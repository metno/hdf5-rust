use std::ptr::{self, addr_of_mut};
use std::slice;
use std::sync::LazyLock;

use hdf5_sys::h5p::{H5Pget_chunk, H5Pget_filter_by_id2, H5Pmodify_filter};
use hdf5_sys::h5t::{H5Tclose, H5Tget_class, H5Tget_size, H5Tget_super, H5T_FLOAT};
use hdf5_sys::h5z::{H5Z_class2_t, H5Z_filter_t, H5Zregister, H5Z_CLASS_T_VERS, H5Z_FLAG_REVERSE};

use crate::error::H5ErrorCode;
use crate::globals::{H5E_CALLBACK, H5E_PLIST};
use crate::internal_prelude::*;

use zfp_sys::zfp_stream_set_reversible;
pub use zfp_sys::{
    stream_close, stream_open, zfp_compress, zfp_decompress, zfp_field_1d, zfp_field_2d,
    zfp_field_3d, zfp_field_4d, zfp_field_free, zfp_stream_close, zfp_stream_maximum_size,
    zfp_stream_open, zfp_stream_rewind, zfp_stream_set_accuracy, zfp_stream_set_bit_stream,
    zfp_stream_set_precision, zfp_stream_set_rate, zfp_type_zfp_type_double,
    zfp_type_zfp_type_float,
};

const ZFP_FILTER_NAME: &[u8] = b"zfp\0";
pub const ZFP_FILTER_ID: H5Z_filter_t = 32013;
const ZFP_FILTER_VERSION: c_uint = 1;

// ZFP mode constants
const ZFP_MODE_RATE: c_uint = 1;
const ZFP_MODE_PRECISION: c_uint = 2;
const ZFP_MODE_ACCURACY: c_uint = 3;
const ZFP_MODE_REVERSIBLE: c_uint = 5;

const ZFP_FILTER_INFO: &H5Z_class2_t = &H5Z_class2_t {
    version: H5Z_CLASS_T_VERS as _,
    id: ZFP_FILTER_ID,
    encoder_present: 1,
    decoder_present: 1,
    name: ZFP_FILTER_NAME.as_ptr().cast(),
    can_apply: Some(can_apply_zfp),
    set_local: Some(set_local_zfp),
    filter: Some(filter_zfp),
};

static ZFP_INIT: LazyLock<Result<(), &'static str>> = LazyLock::new(|| {
    let ret = unsafe { H5Zregister((ZFP_FILTER_INFO as *const H5Z_class2_t).cast()) };
    if H5ErrorCode::is_err_code(ret) {
        return Err("Can't register ZFP filter");
    }
    Ok(())
});

pub fn register_zfp() -> Result<(), &'static str> {
    *ZFP_INIT
}

extern "C" fn can_apply_zfp(_dcpl_id: hid_t, type_id: hid_t, _space_id: hid_t) -> i32 {
    let type_class = unsafe { H5Tget_class(type_id) };
    if type_class == H5T_FLOAT {
        1
    } else {
        0
    }
}

extern "C" fn set_local_zfp(dcpl_id: hid_t, type_id: hid_t, _space_id: hid_t) -> herr_t {
    const MAX_NDIMS: usize = 4;
    let mut flags: c_uint = 0;
    let mut nelmts: size_t = 4;
    let mut values: Vec<c_uint> = vec![0; 4];
    let ret = unsafe {
        H5Pget_filter_by_id2(
            dcpl_id,
            ZFP_FILTER_ID,
            addr_of_mut!(flags),
            addr_of_mut!(nelmts),
            values.as_mut_ptr(),
            0,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    if ret < 0 {
        return -1;
    }
    nelmts = nelmts.max(10);
    values[0] = ZFP_FILTER_VERSION;

    let mut chunkdims: Vec<hsize_t> = vec![0; MAX_NDIMS];
    let ndims: c_int = unsafe { H5Pget_chunk(dcpl_id, MAX_NDIMS as _, chunkdims.as_mut_ptr()) };
    if ndims < 0 {
        return -1;
    }
    if ndims > MAX_NDIMS as _ {
        h5err!("ZFP supports up to 4 dimensions", H5E_PLIST, H5E_CALLBACK);
        return -1;
    }

    let typesize: size_t = unsafe { H5Tget_size(type_id) };
    if typesize == 0 {
        return -1;
    }

    values[1] = ndims as c_uint;
    values[2] = typesize as c_uint;
    for i in 0..ndims as usize {
        if i + 3 < values.len() {
            values[i + 3] = chunkdims[i] as c_uint;
        }
    }

    let r = unsafe { H5Pmodify_filter(dcpl_id, ZFP_FILTER_ID, flags, nelmts, values.as_ptr()) };
    if r < 0 {
        -1
    } else {
        1
    }
}

#[derive(Debug)]
struct ZfpConfig {
    pub ndims: c_int,
    pub typesize: size_t,
    pub dims: [size_t; 4],
    pub mode: c_uint,
    pub rate: f64,
    pub precision: u32,
    pub accuracy: f64,
}

impl From<ZfpConfig> for zfp_sys::zfp_config {
    fn from(cfg: ZfpConfig) -> Self {

        let binding_output = match cfg.mode {
            ZFP_MODE_RATE => {
                zfp_sys::zfp_config__bindgen_ty_1 {
                    rate: cfg.rate
                }
            },
            ZFP_MODE_PRECISION => {
                zfp_sys::zfp_config__bindgen_ty_1 {
                    precision: cfg.precision
                }
            },
            ZFP_MODE_ACCURACY => {
                zfp_sys::zfp_config__bindgen_ty_1 {
                    tolerance: cfg.accuracy
                }

            },
            ZFP_MODE_REVERSIBLE => {
                zfp_sys::zfp_config__bindgen_ty_1 {
                    tolerance: cfg.accuracy
                }

            },
            _ => {
                h5err!("Invalid ZFP mode", H5E_PLIST, H5E_CALLBACK);
            }
        };


        zfp_sys::zfp_config {
            mode: cfg.mode,
            arg: binding_output,
        }

    }
}

fn parse_zfp_cdata(cd_nelmts: size_t, cd_values: *const c_uint) -> Option<ZfpConfig> {
    let cdata = unsafe { slice::from_raw_parts(cd_values, cd_nelmts as _) };
    if cdata.len() < 7 {
        h5err!("Invalid ZFP filter configuration", H5E_PLIST, H5E_CALLBACK);
        return None;
    }

    let ndims = cdata[1] as c_int;
    let typesize = cdata[2] as size_t;
    let mut dims = [0; 4];
    for i in 0..(ndims as usize).min(4) {
        dims[i] = cdata[3 + i] as size_t;
    }

    let mode = if cdata.len() > 7 { cdata[7] } else { ZFP_MODE_RATE };
    let param1 = if cdata.len() > 8 { cdata[8] } else { 0 };
    let param2 = if cdata.len() > 9 { cdata[9] } else { 0 };

    let (rate, precision, accuracy) = match mode {
        ZFP_MODE_RATE => {
            let rate = f64::from_bits(((param1 as u64) << 32) | (param2 as u64));
            (rate, 0, 0.0)
        }
        ZFP_MODE_PRECISION => (0.0, param1, 0.0),
        ZFP_MODE_ACCURACY => {
            let accuracy = f64::from_bits(((param1 as u64) << 32) | (param2 as u64));
            (0.0, 0, accuracy)
        }
        ZFP_MODE_REVERSIBLE => (0.0, 0, 0.0),
        _ => {
            h5err!("Invalid ZFP mode", H5E_PLIST, H5E_CALLBACK);
            return None;
        }
    };

    Some(ZfpConfig { ndims, typesize, dims, mode, rate, precision, accuracy })
}

unsafe extern "C" fn filter_zfp(
    flags: c_uint, cd_nelmts: size_t, cd_values: *const c_uint, nbytes: size_t,
    buf_size: *mut size_t, buf: *mut *mut c_void,
) -> size_t {
    let cfg = if let Some(cfg) = parse_zfp_cdata(cd_nelmts, cd_values) {
        cfg
    } else {
        return 0;
    };
    if flags & H5Z_FLAG_REVERSE == 0 {
        unsafe { filter_zfp_compress(&cfg, buf_size, buf) }
    } else {
        unsafe { filter_zfp_decompress(&cfg, nbytes, buf_size, buf) }
    }
}

unsafe fn filter_zfp_compress(
    cfg: &ZfpConfig, buf_size: *mut size_t, buf: *mut *mut c_void,
) -> size_t {
    let zfp_stream = zfp_stream_open(ptr::null_mut());
    if zfp_stream.is_null() {
        h5err!("Failed to open ZFP stream", H5E_PLIST, H5E_CALLBACK);
        return 0;
    }

    match cfg.mode {
        ZFP_MODE_RATE => {
            zfp_stream_set_rate(zfp_stream, cfg.rate, cfg.typesize as _, cfg.ndims as _, 0);
        }
        ZFP_MODE_PRECISION => {
            zfp_stream_set_precision(zfp_stream, cfg.precision);
        }
        ZFP_MODE_ACCURACY => {
            zfp_stream_set_accuracy(zfp_stream, cfg.accuracy);
        }
        ZFP_MODE_REVERSIBLE => zfp_stream_set_reversible(zfp_stream),
        _ => {
            zfp_stream_close(zfp_stream);
            return 0;
        }
    }

    let field = if cfg.typesize == 4 {
        match cfg.ndims {
            1 => zfp_field_1d((*buf).cast(), zfp_type_zfp_type_float, cfg.dims[0]),
            2 => zfp_field_2d((*buf).cast(), zfp_type_zfp_type_float, cfg.dims[0], cfg.dims[1]),
            3 => zfp_field_3d(
                (*buf).cast(),
                zfp_type_zfp_type_float,
                cfg.dims[0],
                cfg.dims[1],
                cfg.dims[2],
            ),
            4 => zfp_field_4d(
                (*buf).cast(),
                zfp_type_zfp_type_float,
                cfg.dims[0],
                cfg.dims[1],
                cfg.dims[2],
                cfg.dims[3],
            ),
            _ => ptr::null_mut(),
        }
    } else {
        match cfg.ndims {
            1 => zfp_field_1d((*buf).cast(), zfp_type_zfp_type_double, cfg.dims[0]),
            2 => zfp_field_2d((*buf).cast(), zfp_type_zfp_type_double, cfg.dims[0], cfg.dims[1]),
            3 => zfp_field_3d(
                (*buf).cast(),
                zfp_type_zfp_type_double,
                cfg.dims[0],
                cfg.dims[1],
                cfg.dims[2],
            ),
            4 => zfp_field_4d(
                (*buf).cast(),
                zfp_type_zfp_type_double,
                cfg.dims[0],
                cfg.dims[1],
                cfg.dims[2],
                cfg.dims[3],
            ),
            _ => ptr::null_mut(),
        }
    };

    if field.is_null() {
        zfp_stream_close(zfp_stream);
        h5err!("Failed to create ZFP field", H5E_PLIST, H5E_CALLBACK);
        return 0;
    }

    let maxsize = zfp_stream_maximum_size(zfp_stream, field);
    let outbuf = libc::malloc(maxsize);
    if outbuf.is_null() {
        zfp_field_free(field);
        zfp_stream_close(zfp_stream);
        h5err!("Can't allocate compression buffer", H5E_PLIST, H5E_CALLBACK);
        return 0;
    }

    let bitstream = stream_open(outbuf.cast(), maxsize);
    zfp_stream_set_bit_stream(zfp_stream, bitstream);
    zfp_stream_rewind(zfp_stream);

    let compressed_size = zfp_compress(zfp_stream, field);

    stream_close(bitstream);
    zfp_field_free(field);
    zfp_stream_close(zfp_stream);

    if compressed_size == 0 {
        libc::free(outbuf);
        h5err!("ZFP compression failed", H5E_PLIST, H5E_CALLBACK);
        return 0;
    }

    libc::free(*buf);
    *buf = outbuf;
    *buf_size = compressed_size;
    compressed_size
}

unsafe fn filter_zfp_decompress(
    cfg: &ZfpConfig, nbytes: size_t, buf_size: *mut size_t, buf: *mut *mut c_void,
) -> size_t {
    let zfp_stream = zfp_stream_open(ptr::null_mut());
    if zfp_stream.is_null() {
        h5err!("Failed to open ZFP stream", H5E_PLIST, H5E_CALLBACK);
        return 0;
    }

    match cfg.mode {
        ZFP_MODE_RATE => {
            zfp_stream_set_rate(zfp_stream, cfg.rate, cfg.typesize as _, cfg.ndims as _, 0);
        }
        ZFP_MODE_PRECISION => {
            zfp_stream_set_precision(zfp_stream, cfg.precision);
        }
        ZFP_MODE_ACCURACY => {
            zfp_stream_set_accuracy(zfp_stream, cfg.accuracy);
        }
        ZFP_MODE_REVERSIBLE => zfp_stream_set_reversible(zfp_stream),
        _ => {
            zfp_stream_close(zfp_stream);
            return 0;
        }
    }

    let mut outbuf_size = cfg.typesize;
    for i in 0..cfg.ndims as usize {
        outbuf_size *= cfg.dims[i];
    }

    let outbuf = libc::malloc(outbuf_size);
    if outbuf.is_null() {
        zfp_stream_close(zfp_stream);
        h5err!("Can't allocate decompression buffer", H5E_PLIST, H5E_CALLBACK);
        return 0;
    }

    let field = if cfg.typesize == 4 {
        match cfg.ndims {
            1 => zfp_field_1d(outbuf.cast(), zfp_type_zfp_type_float, cfg.dims[0]),
            2 => zfp_field_2d(outbuf.cast(), zfp_type_zfp_type_float, cfg.dims[0], cfg.dims[1]),
            3 => zfp_field_3d(
                outbuf.cast(),
                zfp_type_zfp_type_float,
                cfg.dims[0],
                cfg.dims[1],
                cfg.dims[2],
            ),
            4 => zfp_field_4d(
                outbuf.cast(),
                zfp_type_zfp_type_float,
                cfg.dims[0],
                cfg.dims[1],
                cfg.dims[2],
                cfg.dims[3],
            ),
            _ => ptr::null_mut(),
        }
    } else {
        match cfg.ndims {
            1 => zfp_field_1d(outbuf.cast(), zfp_type_zfp_type_double, cfg.dims[0]),
            2 => zfp_field_2d(outbuf.cast(), zfp_type_zfp_type_double, cfg.dims[0], cfg.dims[1]),
            3 => zfp_field_3d(
                outbuf.cast(),
                zfp_type_zfp_type_double,
                cfg.dims[0],
                cfg.dims[1],
                cfg.dims[2],
            ),
            4 => zfp_field_4d(
                outbuf.cast(),
                zfp_type_zfp_type_double,
                cfg.dims[0],
                cfg.dims[1],
                cfg.dims[2],
                cfg.dims[3],
            ),
            _ => ptr::null_mut(),
        }
    };

    if field.is_null() {
        libc::free(outbuf);
        zfp_stream_close(zfp_stream);
        h5err!("Failed to create ZFP field", H5E_PLIST, H5E_CALLBACK);
        return 0;
    }

    let bitstream = stream_open((*buf).cast(), nbytes);
    zfp_stream_set_bit_stream(zfp_stream, bitstream);
    zfp_stream_rewind(zfp_stream);

    let status = zfp_decompress(zfp_stream, field);

    stream_close(bitstream);
    zfp_field_free(field);
    zfp_stream_close(zfp_stream);

    if status == 0 {
        libc::free(outbuf);
        h5err!("ZFP decompression failed", H5E_PLIST, H5E_CALLBACK);
        return 0;
    }

    libc::free(*buf);
    *buf = outbuf;
    *buf_size = outbuf_size;
    outbuf_size
}
