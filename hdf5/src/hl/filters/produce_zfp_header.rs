use std::ptr::{self, addr_of_mut};
use std::slice;
use std::sync::LazyLock;

use hdf5_sys::h5p::{H5Pget_chunk, H5Pget_filter_by_id2, H5Pmodify_filter};
use hdf5_sys::h5t::{H5Tclose, H5Tget_class, H5Tget_size, H5Tget_super, H5T_FLOAT};
use hdf5_sys::h5z::{H5Z_class2_t, H5Z_filter_t, H5Zregister, H5Z_CLASS_T_VERS, H5Z_FLAG_REVERSE};
use std::mem;


use crate::error::H5ErrorCode;
use crate::globals::{H5E_CALLBACK, H5E_PLIST};
use crate::internal_prelude::*;


#[cfg(feature = "zfp")]
use zfp_sys;

// Import from zfp-sys (or your FFI bindings)
#[cfg(feature = "zfp")]
use zfp_sys::{zfp_field_metadata,zfp_stream_rewind,ZFP_VERSION_PATCH,ZFP_VERSION_MINOR,ZFP_VERSION_MAJOR,zfp_type_zfp_type_float,zfp_field_1d,zfp_field_4d, zfp_field_2d,zfp_field_3d,zfp_stream,zfp_stream_open,zfp_stream_set_mode,zfp_stream_set_reversible,zfp_stream_maximum_size, zfp_write_header,bitstream,stream_open as bs_open,stream_close as bs_close,zfp_field_free,zfp_stream_close,ZFP_VERSION_TWEAK};


const ZFP_FILTER_NAME: &[u8] = b"zfp\0";
pub const ZFP_FILTER_ID: H5Z_filter_t = 32013;
const ZFP_FILTER_VERSION: c_uint = 1;


const H5Z_FILTER_ZFP_VERSION_MAJOR: u32 = 1;
const H5Z_FILTER_ZFP_VERSION_MINOR: u32 = 0;
const H5Z_FILTER_ZFP_VERSION_PATCH: u32 = 0;


// ZFP mode constants
const ZFP_MODE_RATE: c_uint = 1;
const ZFP_MODE_PRECISION: c_uint = 2;
const ZFP_MODE_ACCURACY: c_uint = 3;
const ZFP_MODE_REVERSIBLE: c_uint = 5;

const H5Z_ZFP_CD_NELMTS_MAX: usize = 8; // whatever the header says; set correctly.
fn build_reversible_zfp_cd_values(
    dims: &[usize],   // e.g. &[z, y, x]
    dtype_size: usize // e.g. 4 for float, 8 for double, or use ints
) -> Option<Vec<u32>> {

    dbg!(dims);
    unsafe {
        // 1. Create a ZFP field for the given dimensions and data type
        let field = match dims.len() {
            1 => {
                // zfp_field_1d
                // unimplemented in bindings example; assume exists
                zfp_field_1d(ptr::null_mut(), zfp_type_zfp_type_float, dims[0])
            }
            2 => zfp_field_2d(ptr::null_mut(), zfp_type_zfp_type_float, dims[1], dims[0]),
            3 => zfp_field_3d(ptr::null_mut(), zfp_type_zfp_type_float, dims[2], dims[1], dims[0]),
            4 => zfp_field_4d(ptr::null_mut(), zfp_type_zfp_type_float, dims[3], dims[2], dims[1], dims[0]),
            _ => return None,
        };
        dbg!(&field);
        if field.is_null() {
            return None;
        }

        // 2. Create a ZFP stream
        let zstr = zfp_stream_open(ptr::null_mut());
        dbg!(&zstr);
        if zstr.is_null() {
            zfp_field_free(field);
            return None;
        }

        // 3. Set stream to reversible mode
        zfp_stream_set_reversible(zstr);

        // 4. Compute max header/bitstream buffer size
        let max_bytes = 160;


        // 5. Allocate a buffer to hold the header + data (bitstream)
        let mut buf: Vec<u8> = vec![0u8; max_bytes];
        let buf_ptr = buf.as_mut_ptr() as *mut c_void;



        let buf_size = max_bytes;

        // 6. Open a bitstream in that buffer
        let bstr = bs_open(buf_ptr, buf_size);
        dbg!(&bstr);
        if bstr.is_null() {
            zfp_stream_close(zstr);
            zfp_field_free(field);
            return None;
        }

        // 7. Attach bitstream to zfp stream
        zfp_sys::zfp_stream_set_bit_stream(zstr, bstr);

        let meta = zfp_sys::zfp_field_metadata(field);
        dbg!(meta);



        // 8. Write the ZFP header
        let bits = zfp_write_header(zstr, field, zfp_sys::ZFP_HEADER_FULL);
        dbg!(&bits);
        if bits == 0 {
            bs_close(bstr);
            zfp_stream_close(zstr);
            zfp_field_free(field);
            return None;
        }

        // compute how many bytes used
        let header_bytes = 1 + ((bits as usize - 1) / 8);
        let hdr_cd_nelmts = 1 + ((header_bytes - 1) / std::mem::size_of::<u32>());
        // round up to multiple of u32 (4 bytes)
        let num_u32 = (header_bytes + 3) / 4;

        println!("header_bytes = {}", header_bytes);
        println!("raw header (hex): {:02X?}", &buf[..header_bytes]);



        // 9. Interpret the buffer as u32 CD values (little-endian)
        // Safety: we wrote at least header_bytes; we treat as u32
        buf.set_len(hdr_cd_nelmts * 4);
        let u32slice: &[u32] = slice::from_raw_parts(buf_ptr as *const u32, num_u32);
        dbg!(&buf);

        let cd_values = u32slice.to_vec();

        // 10. Clean up
        bs_close(bstr);
        zfp_stream_close(zstr);
        zfp_field_free(field);


        // get the version header right
        const ZFP_VERSION_NO: u32 =
            (ZFP_VERSION_MAJOR << 12)
                | (ZFP_VERSION_MINOR << 8)
                | (ZFP_VERSION_PATCH << 4)
                | (ZFP_VERSION_TWEAK);

        const H5Z_FILTER_ZFP_VERSION_NO: u32 =
            (H5Z_FILTER_ZFP_VERSION_MAJOR << 8)
                | (H5Z_FILTER_ZFP_VERSION_MINOR << 4)
                | (H5Z_FILTER_ZFP_VERSION_PATCH);


        let first = (ZFP_VERSION_NO << 16) | (5 << 12) | H5Z_FILTER_ZFP_VERSION_NO;
        let mut cd = vec![ first as u32 ];
        cd.extend(&cd_values);

        Some(cd)
    }
}


#[cfg(test)]
mod test{
    use super::{build_reversible_zfp_cd_values};
    use crate::hl::filters::ZfpMode;

    #[test]
    fn reproduce_zfp_header_from_params(){

        let dataset_size = 10000;
        let n_dims = 1;
        let zfp_mode = ZfpMode::Reversible;
        let zfp_config_vector = [5,0,0];
        let cd_values = build_reversible_zfp_cd_values(&[1,10000], 4);
        dbg!(cd_values);
        assert_eq!(1,0);

    }



}