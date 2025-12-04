use std::ptr::{self, addr_of_mut};
use std::slice;
use std::sync::LazyLock;

use hdf5_sys::h5p::{H5Pget_chunk, H5Pget_filter_by_id2, H5Pmodify_filter};
use hdf5_sys::h5t::{H5Tclose, H5Tget_class, H5Tget_size, H5Tget_super, H5T_FLOAT};
use hdf5_sys::h5z::{H5Z_class2_t, H5Z_filter_t, H5Zregister, H5Z_CLASS_T_VERS, H5Z_FLAG_REVERSE};

use crate::error::H5ErrorCode;
use crate::globals::{H5E_CALLBACK, H5E_PLIST};
use crate::internal_prelude::*;

pub use zfp_sys::{
    zfp_field_alloc,zfp_read_header,zfp_field_dimensionality,zfp_field_size,zfp_field_type,zfp_mode,zfp_mode_zfp_mode_fixed_accuracy,zfp_mode_zfp_mode_fixed_precision,zfp_mode_zfp_mode_fixed_rate,zfp_stream_compression_mode,zfp_stream_accuracy,zfp_stream_rate,zfp_stream_precision,stream_close, stream_open, zfp_compress, zfp_field_metadata,zfp_decompress, zfp_field_1d, zfp_field_2d,bitstream,zfp_stream_set_reversible,
    zfp_field_3d, zfp_field_4d, zfp_field_free, zfp_stream_close, zfp_stream_maximum_size,zfp_stream_flush,
    zfp_stream_open, zfp_stream_rewind, zfp_stream_set_accuracy, zfp_stream_set_bit_stream,ZFP_VERSION_MINOR,ZFP_VERSION_PATCH,
    zfp_stream_set_precision, zfp_stream_set_rate, zfp_type_zfp_type_double,zfp_type,ZFP_HEADER_FULL,ZFP_VERSION_MAJOR,ZFP_VERSION_TWEAK,
    zfp_type_zfp_type_float,ZFP_HEADER_MAGIC,ZFP_HEADER_MAX_BITS,ZFP_HEADER_META,ZFP_HEADER_MODE,zfp_write_header,zfp_codec_version,zfp_library_version,zfp_field
};
use zfp_sys::zfp_stream;

use crate::filters::ZfpMode;


/// Major edits are needed to be in alignmeht with the H5Z-ZFP. What was previously implemented was effectively a new implementation of H5Z_ZFP but was incompatible with any library built against it. This reults in bad c_data vectors being created and produces erratic behavior.


const ZFP_FILTER_NAME: &[u8] = b"zfp\0";
pub const ZFP_FILTER_ID: H5Z_filter_t = 32013;
const ZFP_FILTER_VERSION: c_uint = 1;

// ZFP mode constants
const ZFP_MODE_RATE: c_uint = 2;
const ZFP_MODE_PRECISION: c_uint = 3;
const ZFP_MODE_ACCURACY: c_uint = 4;
const ZFP_MODE_REVERSIBLE: c_uint = 5;
const ZFP_MODE_EXPERT: c_uint = 1;


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
    // start with a small buffer; H5Pget_filter_by_id2 will return the stored cdata (mode/params)
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
    // Preserve original small cdata (mode/params) returned by H5Pget_filter_by_id2.
    let orig = values.clone();
    // ensure we have enough space for header + dims + parameters (we need at least indices up to 9)
    nelmts = nelmts.max(10);
    values.resize(nelmts as usize, 0);
    // set version and header entries
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

    // fill header fields (ndims, typesize) and chunk dimensions
    values[1] = ndims as c_uint;
    values[2] = typesize as c_uint;
    for i in 0..(ndims as usize).min(values.len().saturating_sub(3)) {
        values[i + 3] = chunkdims[i] as c_uint;
    }
    // The Filter::apply_zfp() originally stored mode/param1/param2 at indices 0..2.
    // parse_zfp expects these at indices 7..9 in the final cdata layout. Move/preserve them.
    if values.len() >= 10 {
        values[7] = orig.get(0).copied().unwrap_or(0);
        values[8] = orig.get(1).copied().unwrap_or(0);
        values[9] = orig.get(2).copied().unwrap_or(0);

    }
    // temp overrid and changed line 133 to orig instead of values
    let nelmts = 4;

    let r = unsafe { H5Pmodify_filter(dcpl_id, ZFP_FILTER_ID, flags, nelmts, orig.as_ptr()) };
    if r < 0 {
        -1
    } else {
        1
    }
}


fn pack_header_into_cd_values(
    header_bytes: &[u8],
    bits_written: usize,
) -> Vec<u32> {
    let total_bytes = (bits_written + 7) / 8;
    let nwords = (total_bytes + 3) / 4;

    let mut cd_vals = Vec::with_capacity(1 + nwords);

    // cd_values[0] = version word (we'll fill it in below)
    cd_vals.push(0);

    for i in 0..nwords {
        let mut word_bytes = [0u8; 4];
        for j in 0..4 {
            let idx = i * 4 + j;
            if idx < total_bytes {
                word_bytes[j] = header_bytes[idx];
            }
        }
        cd_vals.push(u32::from_le_bytes(word_bytes));
    }

    cd_vals
}


const H5Z_ZFP_CD_NELMTS_MAX: usize = 8; // whatever the header says; set correctly.

pub unsafe fn compute_hdr_cd_values(
    zt: zfp_type,
    ndims_used: usize,
    dims_used: &[u64],
    mode: ZfpMode, // your enum wrapping rate/precision/accuracy/reversible
) -> (Vec<u32>, usize) {
    // 1. Build dummy_field like H5Z_zfp_set_local
    let dummy_field: *mut zfp_field = match ndims_used {
        1 => zfp_field_1d(ptr::null_mut(), zt, dims_used[0].try_into().unwrap()),
        2 => zfp_field_2d(ptr::null_mut(), zt, dims_used[1].try_into().unwrap(), dims_used[0].try_into().unwrap()),
        3 => zfp_field_3d(ptr::null_mut(), zt, dims_used[2].try_into().unwrap(), dims_used[1].try_into().unwrap(), dims_used[0].try_into().unwrap()),
        4 => zfp_field_4d(ptr::null_mut(), zt, dims_used[3].try_into().unwrap(), dims_used[2].try_into().unwrap(), dims_used[1].try_into().unwrap(), dims_used[0].try_into().unwrap()),
        _ => panic!("ZFP supports 1..4 non-unity dims"),
    };
    assert!(!dummy_field.is_null());

    // 2. Prepare the cd_values array like C code: u32 buffer
    let mut hdr_cd_values = vec![0u32; H5Z_ZFP_CD_NELMTS_MAX];

    // 3. Version word (use the macro layout: (ZFP_VERSION_NO<<16)|(ZFP_CODEC<<12)|H5Z_FILTER_ZFP_VERSION_NO)
    hdr_cd_values[0] = make_version_word(); // see previous message
    dbg!(&hdr_cd_values);

    // 4. Treat &hdr_cd_values[1] as bitstream buffer
    let ptr_bytes = hdr_cd_values[1..].as_mut_ptr() as *mut c_void;
    let bytes_len = (hdr_cd_values.len() - 1) * std::mem::size_of::<u32>();

    let dummy_bstr: *mut bitstream = stream_open(ptr_bytes, bytes_len as usize);

    let dummy_zstr: *mut zfp_stream = zfp_stream_open(dummy_bstr);
    // 5. Set mode the same way H5Z_zfp_set_local does
    match mode {
        ZfpMode::Reversible => {
            zfp_stream_set_reversible(dummy_zstr);
        },
    ZfpMode::FixedAccuracy(acc) =>{
        dbg!(&acc);
        zfp_stream_set_accuracy(dummy_zstr,acc);
    },
        // handle Rate/Precision/Accuracy/Expert as needed
        _ => unimplemented!(),
    }

    let field_meta = zfp_sys::zfp_field_metadata(dummy_field);
    dbg!(field_meta);

    // 6. Write FULL header (critical!) into the hdr_cd_values[1..] buffer
    let hdr_bits = zfp_write_header(dummy_zstr, dummy_field, ZFP_HEADER_FULL as u32);
    assert!(hdr_bits != 0);

    // 7. Flush and close (exactly like C)
    zfp_stream_flush(dummy_zstr);
    zfp_stream_close(dummy_zstr);
    stream_close(dummy_bstr);
    zfp_field_free(dummy_field);

    // 8. Compute hdr_bytes/hdr_cd_nelmts as in C
    let hdr_bytes = 1 + ((hdr_bits - 1) / 8);
    let mut hdr_cd_nelmts = 1 + ((hdr_bytes - 1) / std::mem::size_of::<u32>());
    hdr_cd_nelmts += 1; // for slot 0

    (hdr_cd_values, hdr_cd_nelmts)
}

unsafe fn make_version_word() -> u32 {

    // 0xM M P T: for 1.0.0.0 → 0x1000
    const ZFP_VERSION_NO: u32 =
        (ZFP_VERSION_MAJOR << 12)
            | (ZFP_VERSION_MINOR << 8)
            | (ZFP_VERSION_PATCH << 4)
            | (ZFP_VERSION_TWEAK);

    const ZFP_CODEC: u32 = ZFP_VERSION_MINOR; // or 5 if you know you want codec 5

    // Filter version: 1.1.0 → 0x0110
    const H5Z_FILTER_ZFP_VERSION_MAJOR: u32 = 1;
    const H5Z_FILTER_ZFP_VERSION_MINOR: u32 = 1;
    const H5Z_FILTER_ZFP_VERSION_PATCH: u32 = 0;

    const H5Z_FILTER_ZFP_VERSION_NO: u32 =
        (H5Z_FILTER_ZFP_VERSION_MAJOR << 8)
            | (H5Z_FILTER_ZFP_VERSION_MINOR << 4)
            | (H5Z_FILTER_ZFP_VERSION_PATCH);

    // One simple scheme: low 8 bits = codec, high 24 bits = lib version truncated.
    (ZFP_VERSION_NO << 16)
        | (ZFP_CODEC << 12)
        | H5Z_FILTER_ZFP_VERSION_NO
}



pub unsafe fn make_llnl_style_cd_values(
    chunk_dims: &[usize],
    mode: ZfpMode,
) -> Vec<u32> {
    let ztype = zfp_type_zfp_type_float;
    let dims_used: Vec<usize> = chunk_dims
        .iter()
        .copied()
        .filter(|&d| d > 1)
        .collect();
    let field = make_zfp_field(ztype, &dims_used);
    let zfp_stream = zfp_stream_open(ptr::null_mut());


    match mode {
        ZfpMode::FixedRate(rate) => {
            zfp_stream_set_rate(zfp_stream, rate, std::mem::size_of::<f32>() as _, dims_used.len() as _, 0);
        }
        ZfpMode::FixedPrecision(precision) => {
            zfp_stream_set_precision(zfp_stream, precision as u32);
        }
        ZfpMode::FixedAccuracy(accuracy) => {
            zfp_stream_set_accuracy(zfp_stream, accuracy);
        }
        ZfpMode::Reversible => {
            zfp_stream_set_reversible(zfp_stream);
        }
    };


    let (header_bytes, bits_written) = zfp_header_bits(zfp_stream, field);
    dbg!(&header_bytes);
    dbg!(&bits_written);
    let mut cd_vals = pack_header_into_cd_values(&header_bytes, bits_written);

    // plug in version info
    cd_vals[0] = make_version_word();
    dbg!(&cd_vals);
    unsafe {
        zfp_field_free(field);
        zfp_stream_close(zfp_stream);
    }

    cd_vals
}


unsafe fn make_zfp_field(ztype: zfp_type, dims: &[usize]) -> *mut zfp_field {
    let mut shape = [1usize; 4];
    for (idx, dim) in dims.iter().copied().take(4).enumerate() {
        shape[idx] = dim.max(1);
    }

    match dims.len() {
        0 | 1 => zfp_field_1d(ptr::null_mut(), ztype, shape[0]),
        2 => zfp_field_2d(ptr::null_mut(), ztype, shape[0], shape[1]),
        3 => zfp_field_3d(ptr::null_mut(), ztype, shape[0], shape[1], shape[2]),
        _ => zfp_field_4d(
            ptr::null_mut(),
            ztype,
            shape[0],
            shape[1],
            shape[2],
            shape[3],
        ),
    }
}


unsafe fn zfp_header_bits(
    zstream: *mut zfp_stream,
    field: *mut zfp_field,
) -> (Vec<u8>, usize) {
    // Max bits → bytes; this constant is in the ZFP docs and exposed via zfp-sys
    let max_bits = ZFP_HEADER_MAX_BITS as usize;
    let max_bytes = 20;
    let mut buf = vec![0u8; max_bytes];

    // Make bitstream over our byte buffer
    let bs = stream_open(
        buf.as_mut_ptr() as *mut c_void,
        max_bytes as usize,
    );
    zfp_stream_set_bit_stream(zstream, bs);
    zfp_stream_rewind(zstream);

    let mask = ZFP_HEADER_FULL as u32;
    let bits_written = zfp_write_header(zstream, field, mask);

    if bits_written == 0 {
        panic!("zfp_write_header failed");
    }

    // Clean up bitstream; keep header bytes in buf
    stream_close(bs);

    (buf, bits_written as usize)
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


/// receive the new cdata from the system and decode it to recover the right ZFP opertaing modes
pub unsafe fn parse_zfp_cdata(
    cd_nelmts: usize,
    cd_values: *const c_uint,
) -> Option<ZfpConfig> {
    if cd_nelmts < 2 || cd_values.is_null() {
        return None;
    }

    // Full cd array from HDF5: [version_word, header_words...]
    let cdata: &[u32] = slice::from_raw_parts(cd_values, cd_nelmts);

    // We currently ignore the version word, but you can decode it if you want.
    let _version_word = cdata[0];

    // Everything after index 0 is the ZFP header bitstream.
    let header_words = &cdata[1..];
    if header_words.is_empty() {
        return None;
    }

    // Make a mutable copy so we can endian-swap in place if needed.
    let mut header_copy: Vec<u32> = header_words.to_vec();
    let header_bytes = header_copy.len() * std::mem::size_of::<u32>();

    // 1. Open bitstream on the header buffer (like get_zfp_info_from_cd_values) :contentReference[oaicite:2]{index=2}
    let bstr: *mut bitstream =
        stream_open(header_copy.as_mut_ptr() as *mut c_void, header_bytes);
    if bstr.is_null() {
        return None;
    }

    // 2. Open zfp_stream on that bitstream
    let zstr: *mut zfp_stream = zfp_stream_open(bstr);
    if zstr.is_null() {
        stream_close(bstr);
        return None;
    }

    // 3. Allocate a field for header metadata
    let zfld: *mut zfp_field = zfp_field_alloc();
    if zfld.is_null() {
        zfp_stream_close(zstr);
        stream_close(bstr);
        return None;
    }

    // 4. First read only MAGIC, to detect endian or codec mismatch
    let mut bits = zfp_read_header(zstr, zfld, ZFP_HEADER_MAGIC);
    if bits == 0 {
        // Possible endian mismatch: byte-swap each u32 and retry.
        for w in &mut header_copy {
            *w = w.swap_bytes();
        }

        zfp_stream_rewind(zstr);
        bits = zfp_read_header(zstr, zfld, ZFP_HEADER_MAGIC);
        if bits == 0 {
            // Codec mismatch or truly invalid header.
            zfp_field_free(zfld);
            zfp_stream_close(zstr);
            stream_close(bstr);
            return None;
        }
    }

    // 5. We know magic is fine. Rewind and read the full header.
    zfp_stream_rewind(zstr);
    if zfp_read_header(zstr, zfld, ZFP_HEADER_FULL) == 0 {
        zfp_field_free(zfld);
        zfp_stream_close(zstr);
        stream_close(bstr);
        return None;
    }

    // 6. Extract array metadata via high-level API (no manual bit-twiddling).
    let ndims = zfp_field_dimensionality(zfld) as i32;

    // zfp_field_size can fill per-dimension sizes if we pass a buffer.
    let mut size_per_dim: [usize; 4] = [0; 4];
    if ndims > 0 {
        // zfp_field_size returns total number of elements and optionally fills size[i].
        // The C signature uses size_t*; we just alias &mut [usize] here.
        zfp_field_size(
            zfld,
            size_per_dim.as_mut_ptr() as *mut _,
        );
    }

    let mut dims: [usize; 4] = [0; 4];
    for i in 0..(ndims as usize).min(4) {
        dims[i] = size_per_dim[i];
    }

    // Scalar type → element size in bytes.
    let zt: zfp_type = zfp_field_type(zfld);
    let typesize: usize = match zt {
        // Adjust these to match the actual enum variants in your bindings
        x if x == zfp_sys::zfp_type_zfp_type_int32 => std::mem::size_of::<i32>(),
        x if x == zfp_sys::zfp_type_zfp_type_int64 => std::mem::size_of::<i64>(),
        x if x == zfp_sys::zfp_type_zfp_type_float => std::mem::size_of::<f32>(),
        x if x == zfp_sys::zfp_type_zfp_type_double => std::mem::size_of::<f64>(),
        _ => {
            zfp_field_free(zfld);
            zfp_stream_close(zstr);
            stream_close(bstr);
            return None;
        }
    };
    // 7. Extract compression mode and parameters from the stream itself.
    let zmode_enum: zfp_mode = zfp_stream_compression_mode(zstr);
    let mode = zmode_enum as u32;
    dbg!(&mode);

    let mut rate: f64 = 0.0;
    let mut precision: u32 = 0;
    let mut accuracy: f64 = 0.0;

    // These getters are available on modern zfp (1.0+).
    // If your zfp version is older, you can `cfg`-gate or leave them at zero.
    match zmode_enum {
        m if m == zfp_sys::zfp_mode_zfp_mode_fixed_rate => {
            rate = zfp_stream_rate(zstr, ndims as u32);
        }
        m if m == zfp_sys::zfp_mode_zfp_mode_fixed_precision => {
            precision = zfp_stream_precision(zstr);
        }
        m if m == zfp_sys::zfp_mode_zfp_mode_fixed_accuracy => {
            accuracy = zfp_stream_accuracy(zstr);
        },
        m if m == zfp_sys::zfp_mode_zfp_mode_reversible => {
            // no params needed
        }

        // Expert or reversible -> we don’t have a single scalar parameter to expose
        _ => {
            // leave rate/precision/accuracy at 0; you can later extend this by
            // calling zfp_stream_params() for expert mode.
        }
    }
    dbg!(&accuracy);

    // 8. Cleanup
    zfp_field_free(zfld);
    zfp_stream_close(zstr);
    stream_close(bstr);

    Some(ZfpConfig {
        ndims,
        typesize,
        dims,
        mode,
        rate,
        precision,
        accuracy,
    })
}

fn parse_zfp_cdata_old(cd_nelmts: size_t, cd_values: *const c_uint) -> Option<ZfpConfig> {
    let cdata = unsafe { slice::from_raw_parts(cd_values, cd_nelmts as _) };
    dbg!(&cdata);
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
    dbg!(&cfg);

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
            dbg!(cfg.accuracy);
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
    println!("Here Outbuf");

    let bitstream = stream_open(outbuf.cast(), maxsize);
    zfp_stream_set_bit_stream(zfp_stream, bitstream);
    zfp_stream_rewind(zfp_stream);

    let compressed_size = zfp_compress(zfp_stream, field);
    println!("here compressed size");
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
