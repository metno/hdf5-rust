use std::ptr::{self, addr_of_mut};
use std::slice;
use std::sync::LazyLock;

use hdf5_sys::h5p::{H5Pget_chunk, H5Pget_filter_by_id2, H5Pmodify_filter};
use hdf5_sys::h5t::{H5Tclose, H5Tget_class, H5Tget_size, H5Tget_super, H5T_FLOAT};
use hdf5_sys::h5z::{H5Z_class2_t, H5Z_filter_t, H5Zregister, H5Z_CLASS_T_VERS, H5Z_FLAG_REVERSE};

use crate::error::H5ErrorCode;
use crate::globals::{H5E_CALLBACK, H5E_PLIST};
use crate::internal_prelude::*;

use zfp_sys::zfp_stream;
pub use zfp_sys::{
    bitstream, stream_close, stream_open, zfp_codec_version, zfp_compress, zfp_decompress,
    zfp_field, zfp_field_1d, zfp_field_2d, zfp_field_3d, zfp_field_4d, zfp_field_alloc,
    zfp_field_dimensionality, zfp_field_free, zfp_field_metadata, zfp_field_size, zfp_field_type,
    zfp_library_version, zfp_mode, zfp_mode_zfp_mode_fixed_accuracy,
    zfp_mode_zfp_mode_fixed_precision, zfp_mode_zfp_mode_fixed_rate, zfp_read_header,
    zfp_stream_accuracy, zfp_stream_close, zfp_stream_compression_mode, zfp_stream_flush,
    zfp_stream_maximum_size, zfp_stream_open, zfp_stream_precision, zfp_stream_rate,
    zfp_stream_rewind, zfp_stream_set_accuracy, zfp_stream_set_bit_stream,
    zfp_stream_set_precision, zfp_stream_set_rate, zfp_stream_set_reversible, zfp_type,
    zfp_type_zfp_type_double, zfp_type_zfp_type_float, zfp_write_header, ZFP_HEADER_FULL,
    ZFP_HEADER_MAGIC, ZFP_HEADER_MAX_BITS, ZFP_HEADER_META, ZFP_HEADER_MODE, ZFP_VERSION_MAJOR,
    ZFP_VERSION_MINOR, ZFP_VERSION_PATCH, ZFP_VERSION_TWEAK,
};

use crate::filters::ZfpMode;

/// Major edits are needed to be in alignmeht with the H5Z-ZFP. What was previously implemented was
/// effectively a new implementation of H5Z_ZFP but was incompatible with any library built against
/// it. This results in bad c_data vectors being created and produces erratic behavior.

pub(crate) const MAX_NDIMS: usize = 4;

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

/// Sets the local properties for the ZFP filter.
///
/// This function is called during the creation of a dataset or attribute to set
/// the local properties of the ZFP filter. It retrieves the filter's configuration
/// data, validates the chunk dimensions, and updates the filter's parameters.
///
/// # Parameters
/// - `dcpl_id`: The dataset creation property list identifier.
/// - `type_id`: The datatype identifier of the dataset or attribute.
/// - `_space_id`: The dataspace identifier (not used in this function).
///
/// # Returns
/// - `herr_t`: Returns 1 on success, or -1 on failure.
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

const H5Z_ZFP_CD_NELMTS_MAX: usize = 8; // whatever the header says; set correctly.

/// Computes the header and configuration data values for the ZFP filter.
///
/// This function generates the header and configuration data values (`cd_values`)
/// required for the ZFP filter. It creates a dummy ZFP field based on the provided
/// dimensions and data type, sets the compression mode, and writes the full header
/// into the `cd_values` buffer.
///
/// # Parameters
/// - `zt`: The ZFP data type (e.g., `zfp_type_zfp_type_float` or `zfp_type_zfp_type_double`).
/// - `ndims_used`: The number of dimensions used in the data.
/// - `dims_used`: A slice containing the sizes of the dimensions.
/// - `mode`: The ZFP compression mode, which can be fixed rate, precision, accuracy, or reversible.
///
/// # Returns
/// A tuple containing:
/// - `Vec<u32>`: The header and configuration data values.
/// - `usize`: The number of elements in the `cd_values` array.
///
/// # Panics
/// This function will panic if the number of dimensions exceeds the supported range (1 to 4).
pub unsafe fn compute_hdr_cd_values(
    zt: zfp_type,
    ndims_used: usize,
    dims_used: &[u64],
    mode: ZfpMode, // your enum wrapping rate/precision/accuracy/reversible
) -> (Vec<u32>, usize) {
    // 1. Build dummy_field like H5Z_zfp_set_local
    let dummy_field: *mut zfp_field = match ndims_used {
        1 => zfp_field_1d(ptr::null_mut(), zt, dims_used[0].try_into().unwrap()),
        2 => zfp_field_2d(
            ptr::null_mut(),
            zt,
            dims_used[1].try_into().unwrap(),
            dims_used[0].try_into().unwrap(),
        ),
        3 => zfp_field_3d(
            ptr::null_mut(),
            zt,
            dims_used[2].try_into().unwrap(),
            dims_used[1].try_into().unwrap(),
            dims_used[0].try_into().unwrap(),
        ),
        4 => zfp_field_4d(
            ptr::null_mut(),
            zt,
            dims_used[3].try_into().unwrap(),
            dims_used[2].try_into().unwrap(),
            dims_used[1].try_into().unwrap(),
            dims_used[0].try_into().unwrap(),
        ),
        _ => panic!("ZFP supports 1..4 non-unity dims"),
    };
    assert!(!dummy_field.is_null());

    // 2. Prepare the cd_values array like C code: u32 buffer
    let mut hdr_cd_values = vec![0u32; H5Z_ZFP_CD_NELMTS_MAX];

    // 3. Version word (use the macro layout: (ZFP_VERSION_NO<<16)|(ZFP_CODEC<<12)|H5Z_FILTER_ZFP_VERSION_NO)
    hdr_cd_values[0] = make_version_word(); // see previous message

    // 4. Treat &hdr_cd_values[1] as bitstream buffer
    let ptr_bytes = hdr_cd_values[1..].as_mut_ptr() as *mut c_void;
    let bytes_len = (hdr_cd_values.len() - 1) * std::mem::size_of::<u32>();

    let dummy_bstr: *mut bitstream = stream_open(ptr_bytes, bytes_len as usize);

    let dummy_zstr: *mut zfp_stream = zfp_stream_open(dummy_bstr);
    // 5. Set mode the same way H5Z_zfp_set_local does
    match mode {
        ZfpMode::Reversible => {
            zfp_stream_set_reversible(dummy_zstr);
        }
        ZfpMode::FixedAccuracy(acc) => {
            zfp_stream_set_accuracy(dummy_zstr, acc);
        }

        ZfpMode::FixedRate(rate) => {
            zfp_stream_set_rate(dummy_zstr, rate, zt, ndims_used as u32, 0);
        }
        ZfpMode::FixedPrecision(precision) => {
            zfp_stream_set_precision(dummy_zstr, precision as u32);
        }
        // handle Rate/Precision/Accuracy/Expert as needed
        _ => unimplemented!(),
    }

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

/// Constructs a version word for the ZFP filter.
///
/// This function generates a 32-bit version word that encodes the ZFP library version,
/// codec version, and filter version. The version word is structured as follows:
/// - High 24 bits: ZFP library version (major, minor, patch, tweak).
/// - Middle 8 bits: Codec version.
/// - Low 8 bits: Filter version.
///
/// # Returns
/// A 32-bit unsigned integer representing the version word.
unsafe fn make_version_word() -> u32 {
    // 0xM M P T: for 1.0.0.0 → 0x1000
    const ZFP_VERSION_NO: u32 = (ZFP_VERSION_MAJOR << 12)
        | (ZFP_VERSION_MINOR << 8)
        | (ZFP_VERSION_PATCH << 4)
        | (ZFP_VERSION_TWEAK);

    const ZFP_CODEC: u32 = ZFP_VERSION_MINOR; // or 5 if you know you want codec 5

    // Filter version: 1.1.0 → 0x0110
    const H5Z_FILTER_ZFP_VERSION_MAJOR: u32 = 1;
    const H5Z_FILTER_ZFP_VERSION_MINOR: u32 = 1;
    const H5Z_FILTER_ZFP_VERSION_PATCH: u32 = 0;

    const H5Z_FILTER_ZFP_VERSION_NO: u32 = (H5Z_FILTER_ZFP_VERSION_MAJOR << 8)
        | (H5Z_FILTER_ZFP_VERSION_MINOR << 4)
        | (H5Z_FILTER_ZFP_VERSION_PATCH);

    // One simple scheme: low 8 bits = codec, high 24 bits = lib version truncated.
    (ZFP_VERSION_NO << 16) | (ZFP_CODEC << 12) | H5Z_FILTER_ZFP_VERSION_NO
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

/// Parses ZFP filter configuration data from the given input.
///
/// This function extracts metadata and compression parameters from the
/// provided `cd_values` array, which represents the ZFP filter's configuration
/// data. It handles endian mismatches, validates the header, and retrieves
/// information such as dimensions, data type, and compression mode.
///
/// # Safety
/// This function is marked as unsafe because it performs raw pointer
/// dereferencing and interacts with low-level C APIs, which require careful
/// handling to avoid undefined behavior.
///
/// # Parameters
/// - `cd_nelmts`: The number of elements in the `cd_values` array.
/// - `cd_values`: A pointer to the array of configuration data values.
///
/// # Returns
/// - `Option<ZfpConfig>`: Returns a `ZfpConfig` struct containing the parsed
///   metadata and compression parameters if successful, or `None` if the
///   parsing fails.
pub unsafe fn parse_zfp_cdata(cd_nelmts: usize, cd_values: *const c_uint) -> Option<ZfpConfig> {
    if cd_nelmts < 2 || cd_values.is_null() {
        return None;
    }

    // Full cd array from HDF5: [version_word, header_words...]
    let cdata: &[u32] = slice::from_raw_parts(cd_values, cd_nelmts);

    // ignore the version word,
    let _version_word = cdata[0];

    // ZFP header bitstream.
    let header_words = &cdata[1..];
    if header_words.is_empty() {
        return None;
    }

    // Make a mutable copy so we can endian-swap in place if needed.
    let mut header_copy: Vec<u32> = header_words.to_vec();
    let header_bytes = header_copy.len() * std::mem::size_of::<u32>();

    // Open bitstream on the header buffer (like get_zfp_info_from_cd_values)
    let bstr: *mut bitstream = stream_open(header_copy.as_mut_ptr() as *mut c_void, header_bytes);
    if bstr.is_null() {
        return None;
    }

    // Open zfp_stream on that bitstream
    let zstr: *mut zfp_stream = zfp_stream_open(bstr);
    if zstr.is_null() {
        stream_close(bstr);
        return None;
    }

    // Allocate a field for header metadata
    let zfld: *mut zfp_field = zfp_field_alloc();
    if zfld.is_null() {
        zfp_stream_close(zstr);
        stream_close(bstr);
        return None;
    }

    //First read only MAGIC, to detect endian or codec mismatch
    let mut bits = zfp_read_header(zstr, zfld, ZFP_HEADER_MAGIC);
    if bits == 0 {
        // Possible endian mismatch: byte-swap each u32 and retry.
        for w in &mut header_copy {
            *w = w.swap_bytes();
        }

        zfp_stream_rewind(zstr);
        bits = zfp_read_header(zstr, zfld, ZFP_HEADER_MAGIC);
        if bits == 0 {
            zfp_field_free(zfld);
            zfp_stream_close(zstr);
            stream_close(bstr);
            return None;
        }
    }

    //Rewind and read the full header.
    zfp_stream_rewind(zstr);
    if zfp_read_header(zstr, zfld, ZFP_HEADER_FULL) == 0 {
        zfp_field_free(zfld);
        zfp_stream_close(zstr);
        stream_close(bstr);
        return None;
    }

    //Extract array metadata
    let ndims = zfp_field_dimensionality(zfld) as i32;

    // zfp_field_size can fill per-dimension sizes; pass a buffer.
    let mut size_per_dim: [usize; 4] = [0; 4];
    if ndims > 0 {
        // zfp_field_size returns total number of elements and optionally fills size[i].
        // The C signature uses size_t*;  just alias &mut [usize] here.
        zfp_field_size(zfld, size_per_dim.as_mut_ptr() as *mut _);
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
    // Extract compression mode and parameters from the stream itself.
    let zmode_enum: zfp_mode = zfp_stream_compression_mode(zstr);
    let mode = zmode_enum as u32;

    let mut rate: f64 = 0.0;
    let mut precision: u32 = 0;
    let mut accuracy: f64 = 0.0;

    // These getters are available on modern zfp (1.0+).
    match zmode_enum {
        m if m == zfp_sys::zfp_mode_zfp_mode_fixed_rate => {
            rate = zfp_stream_rate(zstr, ndims as u32);
        }
        m if m == zfp_sys::zfp_mode_zfp_mode_fixed_precision => {
            precision = zfp_stream_precision(zstr);
        }
        m if m == zfp_sys::zfp_mode_zfp_mode_fixed_accuracy => {
            accuracy = zfp_stream_accuracy(zstr);
        }
        m if m == zfp_sys::zfp_mode_zfp_mode_reversible => {
            // no params needed
        }

        // Expert or reversible -> we don’t have a single scalar parameter to expose
        _ => {}
    }

    //Cleanup
    zfp_field_free(zfld);
    zfp_stream_close(zstr);
    stream_close(bstr);

    Some(ZfpConfig { ndims, typesize, dims, mode, rate, precision, accuracy })
}

/// Applies the ZFP filter for compression or decompression.
///
/// This function serves as the entry point for the ZFP filter, determining whether
/// to compress or decompress the data based on the provided flags. It parses the
/// filter configuration data, validates it, and then delegates the operation to
/// either the compression or decompression function.
///
/// # Safety
/// This function is marked as unsafe because it interacts with raw pointers and
/// performs operations that require careful handling to avoid undefined behavior.
///
/// # Parameters
/// - `flags`: A bitmask indicating the operation mode (e.g., compression or decompression).
/// - `cd_nelmts`: The number of elements in the `cd_values` array.
/// - `cd_values`: A pointer to the array of configuration data values.
/// - `nbytes`: The size of the input buffer in bytes.
/// - `buf_size`: A pointer to the size of the output buffer.
/// - `buf`: A pointer to the input/output buffer.
///
/// # Returns
/// - `size_t`: The size of the processed data (compressed or decompressed) on success,
///   or 0 on failure.
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
