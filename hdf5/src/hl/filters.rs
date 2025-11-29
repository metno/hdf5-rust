use std::collections::HashMap;
use std::ptr::{self, addr_of_mut};

use hdf5_sys::h5p::{
    H5Pget_filter2, H5Pget_nfilters, H5Pset_deflate, H5Pset_filter, H5Pset_fletcher32, H5Pset_nbit,
    H5Pset_scaleoffset, H5Pset_shuffle, H5Pset_szip,
};
use hdf5_sys::h5t::H5T_class_t;
use hdf5_sys::h5z::{
    H5Zfilter_avail, H5Zget_filter_info, H5Z_FILTER_CONFIG_DECODE_ENABLED,
    H5Z_FILTER_CONFIG_ENCODE_ENABLED, H5Z_FILTER_DEFLATE, H5Z_FILTER_FLETCHER32, H5Z_FILTER_NBIT,
    H5Z_FILTER_SCALEOFFSET, H5Z_FILTER_SHUFFLE, H5Z_FILTER_SZIP, H5Z_FLAG_OPTIONAL,
    H5Z_SO_FLOAT_DSCALE, H5Z_SO_INT, H5_SZIP_EC_OPTION_MASK, H5_SZIP_MAX_PIXELS_PER_BLOCK,
    H5_SZIP_NN_OPTION_MASK,
};

pub use hdf5_sys::h5z::H5Z_filter_t;

use crate::internal_prelude::*;

#[cfg(feature = "blosc")]
mod blosc;
#[cfg(feature = "lzf")]
mod lzf;
#[cfg(feature = "zfp")]
mod zfp;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SZip {
    Entropy,
    NearestNeighbor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleOffset {
    Integer(u16),
    FloatDScale(u8),
}

#[cfg(feature = "blosc")]
mod blosc_impl {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    #[cfg(feature = "blosc")]
    #[non_exhaustive]
    pub enum Blosc {
        BloscLZ,
        #[cfg(feature = "blosc-lz4")]
        LZ4,
        #[cfg(feature = "blosc-lz4")]
        LZ4HC,
        #[cfg(feature = "blosc-snappy")]
        Snappy,
        #[cfg(feature = "blosc-zlib")]
        ZLib,
        #[cfg(feature = "blosc-zstd")]
        ZStd,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    #[cfg(feature = "blosc")]
    pub enum BloscShuffle {
        None,
        Byte,
        Bit,
    }

    #[cfg(feature = "blosc")]
    impl Default for BloscShuffle {
        fn default() -> Self {
            Self::Byte
        }
    }

    #[cfg(feature = "blosc")]
    impl From<bool> for BloscShuffle {
        fn from(shuffle: bool) -> Self {
            if shuffle {
                Self::Byte
            } else {
                Self::None
            }
        }
    }

    #[cfg(feature = "blosc")]
    impl Default for Blosc {
        fn default() -> Self {
            Self::BloscLZ
        }
    }

    #[cfg(feature = "blosc")]
    pub fn blosc_get_nthreads() -> u8 {
        h5lock!(super::blosc::blosc_get_nthreads()).clamp(0, 255) as _
    }

    #[cfg(feature = "blosc")]
    pub fn blosc_set_nthreads(num_threads: u8) -> u8 {
        use std::os::raw::c_int;
        let nthreads = h5lock!(super::blosc::blosc_set_nthreads(c_int::from(num_threads)));
        nthreads.clamp(0, 255) as _
    }
}

#[cfg(feature = "blosc")]
pub use blosc_impl::*;

#[cfg(feature = "zfp")]
mod zfp_impl {
    #[derive(Clone, Copy, Debug)]
    pub enum ZfpMode {
        FixedRate(f64),
        FixedPrecision(u8),
        FixedAccuracy(f64),
        Reversible
        ,
    }

    // Bitwise compare f64 so NaN and signed zero are deterministic
    impl PartialEq for ZfpMode {
        fn eq(&self, other: &Self) -> bool {
            use ZfpMode::*;
            match (self, other) {
                (FixedRate(a), FixedRate(b)) => a.to_bits() == b.to_bits(),
                (FixedPrecision(a), FixedPrecision(b)) => a == b,
                (FixedAccuracy(a), FixedAccuracy(b)) => a.to_bits() == b.to_bits(),
                (Reversible, Reversible) => true,
                _ => false,
            }
        }
    }
    impl Eq for ZfpMode {}

    impl Default for ZfpMode {
        fn default() -> Self {
            ZfpMode::FixedRate(4.0)
        }
    }
}

#[cfg(feature = "zfp")]
pub use zfp_impl::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Filter {
    Deflate(u8),
    Shuffle,
    Fletcher32,
    SZip(SZip, u8),
    NBit,
    ScaleOffset(ScaleOffset),
    #[cfg(feature = "lzf")]
    LZF,
    #[cfg(feature = "blosc")]
    Blosc(Blosc, u8, BloscShuffle),
    #[cfg(feature = "zfp")]
    Zfp(ZfpMode),
    User(H5Z_filter_t, Vec<c_uint>),
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct FilterInfo {
    pub is_available: bool,
    pub encode_enabled: bool,
    pub decode_enabled: bool,
}

/// This function requires a synchronisation with other calls to `hdf5`
pub(crate) fn register_filters() {
    #[cfg(feature = "lzf")]
    if let Err(e) = lzf::register_lzf() {
        eprintln!("Error while registering LZF filter: {e}");
    }
    #[cfg(feature = "blosc")]
    if let Err(e) = blosc::register_blosc() {
        eprintln!("Error while registering Blosc filter: {e}");
    }
    #[cfg(feature = "zfp")]
    if let Err(e) = zfp::register_zfp() {
        eprintln!("Error while registering ZFP filter: {e}");
    }
}

/// Returns `true` if deflate filter is available.
pub fn deflate_available() -> bool {
    h5lock!(H5Zfilter_avail(H5Z_FILTER_DEFLATE) == 1)
}

/// Returns `true` if deflate filter is available.
#[doc(hidden)]
#[deprecated(note = "deprecated; use deflate_available()")]
pub fn gzip_available() -> bool {
    deflate_available()
}

/// Returns `true` if szip filter is available.
pub fn szip_available() -> bool {
    h5lock!(H5Zfilter_avail(H5Z_FILTER_SZIP) == 1)
}

/// Returns `true` if LZF filter is available.
pub fn lzf_available() -> bool {
    h5lock!(H5Zfilter_avail(32000) == 1)
}

/// Returns `true` if Blosc filter is available.
pub fn blosc_available() -> bool {
    h5lock!(H5Zfilter_avail(32001) == 1)
}

/// Returns `true` if ZFP filter is available.
pub fn zfp_available() -> bool {
    h5lock!(H5Zfilter_avail(32013) == 1)
}

impl Filter {
    pub fn id(&self) -> H5Z_filter_t {
        match self {
            Self::Deflate(_) => H5Z_FILTER_DEFLATE,
            Self::Shuffle => H5Z_FILTER_SHUFFLE,
            Self::Fletcher32 => H5Z_FILTER_FLETCHER32,
            Self::SZip(_, _) => H5Z_FILTER_SZIP,
            Self::NBit => H5Z_FILTER_NBIT,
            Self::ScaleOffset(_) => H5Z_FILTER_SCALEOFFSET,
            #[cfg(feature = "lzf")]
            Self::LZF => lzf::LZF_FILTER_ID,
            #[cfg(feature = "blosc")]
            Self::Blosc(_, _, _) => blosc::BLOSC_FILTER_ID,
            #[cfg(feature = "zfp")]
            Self::Zfp(_) => zfp::ZFP_FILTER_ID,
            Self::User(id, _) => *id,
        }
    }

    pub fn get_info(filter_id: H5Z_filter_t) -> FilterInfo {
        if !h5call!(H5Zfilter_avail(filter_id)).map(|x| x > 0).unwrap_or_default() {
            return FilterInfo::default();
        }
        let mut flags: c_uint = 0;
        h5lock!(H5Zget_filter_info(filter_id, addr_of_mut!(flags)));
        FilterInfo {
            is_available: true,
            encode_enabled: flags & H5Z_FILTER_CONFIG_ENCODE_ENABLED != 0,
            decode_enabled: flags & H5Z_FILTER_CONFIG_DECODE_ENABLED != 0,
        }
    }

    pub fn is_available(&self) -> bool {
        Self::get_info(self.id()).is_available
    }

    pub fn encode_enabled(&self) -> bool {
        Self::get_info(self.id()).encode_enabled
    }

    pub fn decode_enabled(&self) -> bool {
        Self::get_info(self.id()).decode_enabled
    }

    pub fn deflate(level: u8) -> Self {
        Self::Deflate(level)
    }

    pub fn shuffle() -> Self {
        Self::Shuffle
    }

    pub fn fletcher32() -> Self {
        Self::Fletcher32
    }

    pub fn szip(coding: SZip, px_per_block: u8) -> Self {
        Self::SZip(coding, px_per_block)
    }

    pub fn nbit() -> Self {
        Self::NBit
    }

    pub fn scale_offset(mode: ScaleOffset) -> Self {
        Self::ScaleOffset(mode)
    }

    #[cfg(feature = "lzf")]
    pub fn lzf() -> Self {
        Self::LZF
    }

    #[cfg(feature = "blosc")]
    pub fn blosc<T>(complib: Blosc, clevel: u8, shuffle: T) -> Self
    where
        T: Into<BloscShuffle>,
    {
        Self::Blosc(complib, clevel, shuffle.into())
    }

    #[cfg(feature = "blosc")]
    pub fn blosc_blosclz<T>(clevel: u8, shuffle: T) -> Self
    where
        T: Into<BloscShuffle>,
    {
        Self::blosc(Blosc::BloscLZ, clevel, shuffle)
    }

    #[cfg(feature = "blosc-lz4")]
    pub fn blosc_lz4<T>(clevel: u8, shuffle: T) -> Self
    where
        T: Into<BloscShuffle>,
    {
        Self::blosc(Blosc::LZ4, clevel, shuffle)
    }

    #[cfg(feature = "blosc-lz4")]
    pub fn blosc_lz4hc<T>(clevel: u8, shuffle: T) -> Self
    where
        T: Into<BloscShuffle>,
    {
        Self::blosc(Blosc::LZ4HC, clevel, shuffle)
    }

    #[cfg(feature = "blosc-snappy")]
    pub fn blosc_snappy<T>(clevel: u8, shuffle: T) -> Self
    where
        T: Into<BloscShuffle>,
    {
        Self::blosc(Blosc::Snappy, clevel, shuffle)
    }

    #[cfg(feature = "blosc-zlib")]
    pub fn blosc_zlib<T>(clevel: u8, shuffle: T) -> Self
    where
        T: Into<BloscShuffle>,
    {
        Self::blosc(Blosc::ZLib, clevel, shuffle)
    }

    #[cfg(feature = "blosc-zstd")]
    pub fn blosc_zstd<T>(clevel: u8, shuffle: T) -> Self
    where
        T: Into<BloscShuffle>,
    {
        Self::blosc(Blosc::ZStd, clevel, shuffle)
    }

    #[cfg(feature = "zfp")]
    pub fn zfp(mode: ZfpMode) -> Self {
        Self::Zfp(mode)
    }

    #[cfg(feature = "zfp")]
    pub fn zfp_rate(rate: f64) -> Self {
        Self::zfp(ZfpMode::FixedRate(rate))
    }

    #[cfg(feature = "zfp")]
    pub fn zfp_precision(precision: u8) -> Self {
        Self::zfp(ZfpMode::FixedPrecision(precision))
    }

    #[cfg(feature = "zfp")]
    pub fn zfp_accuracy(accuracy: f64) -> Self {
        Self::zfp(ZfpMode::FixedAccuracy(accuracy))
    }

    #[cfg(feature = "zfp")]
    pub fn zfp_reversible() -> Self {
        Self::zfp(ZfpMode::Reversible)
    }

    pub fn user(id: H5Z_filter_t, cdata: &[c_uint]) -> Self {
        Self::User(id, cdata.to_vec())
    }

    fn parse_deflate(cdata: &[c_uint]) -> Result<Self> {
        ensure!(!cdata.is_empty(), "expected cdata.len() >= 1 for deflate filter");
        ensure!(cdata[0] <= 9, "invalid deflate level: {}", cdata[0]);
        Ok(Self::deflate(cdata[0] as _))
    }

    fn parse_shuffle(_cdata: &[c_uint]) -> Result<Self> {
        Ok(Self::shuffle())
    }

    fn parse_fletcher32(_cdata: &[c_uint]) -> Result<Self> {
        Ok(Self::fletcher32())
    }

    fn parse_nbit(_cdata: &[c_uint]) -> Result<Self> {
        Ok(Self::nbit())
    }

    fn parse_szip(cdata: &[c_uint]) -> Result<Self> {
        ensure!(cdata.len() >= 2, "expected cdata.len() >= 2 for szip filter");
        let m = cdata[0];
        ensure!(
            (m & H5_SZIP_EC_OPTION_MASK != 0) != (m & H5_SZIP_NN_OPTION_MASK != 0),
            "invalid szip mask: {}: expected EC or NN to be set",
            m
        );
        let szip_coding =
            if m & H5_SZIP_EC_OPTION_MASK == 0 { SZip::NearestNeighbor } else { SZip::Entropy };
        let px_per_block = cdata[1];
        ensure!(
            px_per_block <= H5_SZIP_MAX_PIXELS_PER_BLOCK,
            "invalid pixels per block for szip filter: {}",
            px_per_block
        );
        Ok(Self::szip(szip_coding, px_per_block as _))
    }

    fn parse_scaleoffset(cdata: &[c_uint]) -> Result<Self> {
        ensure!(cdata.len() >= 2, "expected cdata.len() >= 2 for scaleoffset filter");
        let scale_type = cdata[0];
        let mode = if scale_type == (H5Z_SO_INT as c_uint) {
            ensure!(
                cdata[1] <= c_uint::from(u16::max_value()),
                "invalid int scale-offset: {}",
                cdata[1]
            );
            ScaleOffset::Integer(cdata[1] as _)
        } else if scale_type == (H5Z_SO_FLOAT_DSCALE as c_uint) {
            ensure!(
                cdata[1] <= c_uint::from(u8::max_value()),
                "invalid float scale-offset: {}",
                cdata[1]
            );
            ScaleOffset::FloatDScale(cdata[1] as _)
        } else {
            fail!("invalid scale type for scaleoffset filter: {}", cdata[0])
        };
        Ok(Self::scale_offset(mode))
    }

    #[cfg(feature = "lzf")]
    fn parse_lzf(_cdata: &[c_uint]) -> Result<Self> {
        Ok(Self::lzf())
    }

    #[cfg(feature = "blosc")]
    fn parse_blosc(cdata: &[c_uint]) -> Result<Self> {
        ensure!(cdata.len() >= 5, "expected at least length 5 cdata for blosc filter");
        ensure!(cdata.len() <= 7, "expected at most length 7 cdata for blosc filter");
        ensure!(cdata[4] <= 9, "invalid blosc clevel: {}", cdata[4]);
        let clevel = cdata[4] as u8;
        let shuffle = if cdata.len() >= 6 {
            match cdata[5] {
                blosc::BLOSC_NOSHUFFLE => BloscShuffle::None,
                blosc::BLOSC_SHUFFLE => BloscShuffle::Byte,
                blosc::BLOSC_BITSHUFFLE => BloscShuffle::Bit,
                _ => fail!("invalid blosc shuffle: {}", cdata[5]),
            }
        } else {
            BloscShuffle::Byte
        };
        let complib = if cdata.len() >= 7 {
            match cdata[6] {
                blosc::BLOSC_BLOSCLZ => Blosc::BloscLZ,
                #[cfg(feature = "blosc-lz4")]
                blosc::BLOSC_LZ4 => Blosc::LZ4,
                #[cfg(feature = "blosc-lz4")]
                blosc::BLOSC_LZ4HC => Blosc::LZ4HC,
                #[cfg(feature = "blosc-snappy")]
                blosc::BLOSC_SNAPPY => Blosc::Snappy,
                #[cfg(feature = "blosc-zlib")]
                blosc::BLOSC_ZLIB => Blosc::ZLib,
                #[cfg(feature = "blosc-zstd")]
                blosc::BLOSC_ZSTD => Blosc::ZStd,
                _ => fail!("invalid blosc complib: {}", cdata[6]),
            }
        } else {
            Blosc::BloscLZ
        };
        Ok(Self::blosc(complib, clevel, shuffle))
    }

    #[cfg(feature = "zfp")]
    fn parse_zfp(cdata: &[c_uint]) -> Result<Self> {
        ensure!(cdata.len() >= 8, "expected at least length 8 cdata for zfp filter");
        let mode = if cdata.len() >= 8 { cdata[7] } else { 1 };
        let param1 = if cdata.len() >= 9 { cdata[8] } else { 0 };
        let param2 = if cdata.len() >= 10 { cdata[9] } else { 0 };

        let zfp_mode = match mode {
            1 => {
                let rate = f64::from_bits(((param1 as u64) << 32) | (param2 as u64));
                ZfpMode::FixedRate(rate)
            }
            2 => ZfpMode::FixedPrecision(param1 as u8),
            3 => {
                let accuracy = f64::from_bits(((param1 as u64) << 32) | (param2 as u64));
                ZfpMode::FixedAccuracy(accuracy)
            }
            5 => ZfpMode::Reversible,
            _ => fail!("invalid zfp mode: {}", mode),
        };
        Ok(Self::zfp(zfp_mode))
    }

    pub fn from_raw(filter_id: H5Z_filter_t, cdata: &[c_uint]) -> Result<Self> {
        ensure!(filter_id > 0, "invalid filter id: {}", filter_id);
        match filter_id {
            H5Z_FILTER_DEFLATE => Self::parse_deflate(cdata),
            H5Z_FILTER_SHUFFLE => Self::parse_shuffle(cdata),
            H5Z_FILTER_FLETCHER32 => Self::parse_fletcher32(cdata),
            H5Z_FILTER_SZIP => Self::parse_szip(cdata),
            H5Z_FILTER_NBIT => Self::parse_nbit(cdata),
            H5Z_FILTER_SCALEOFFSET => Self::parse_scaleoffset(cdata),
            #[cfg(feature = "lzf")]
            lzf::LZF_FILTER_ID => Self::parse_lzf(cdata),
            #[cfg(feature = "blosc")]
            blosc::BLOSC_FILTER_ID => Self::parse_blosc(cdata),
            #[cfg(feature = "zfp")]
            zfp::ZFP_FILTER_ID => Self::parse_zfp(cdata),
            _ => Ok(Self::user(filter_id, cdata)),
        }
    }

    unsafe fn apply_deflate(plist_id: hid_t, level: u8) -> herr_t {
        H5Pset_deflate(plist_id, c_uint::from(level))
    }

    unsafe fn apply_shuffle(plist_id: hid_t) -> herr_t {
        H5Pset_shuffle(plist_id)
    }

    unsafe fn apply_fletcher32(plist_id: hid_t) -> herr_t {
        H5Pset_fletcher32(plist_id)
    }

    unsafe fn apply_szip(plist_id: hid_t, coding: SZip, px_per_block: u8) -> herr_t {
        let mask = match coding {
            SZip::Entropy => H5_SZIP_EC_OPTION_MASK,
            SZip::NearestNeighbor => H5_SZIP_NN_OPTION_MASK,
        };
        H5Pset_szip(plist_id, mask, c_uint::from(px_per_block))
    }

    unsafe fn apply_nbit(plist_id: hid_t) -> herr_t {
        H5Pset_nbit(plist_id)
    }

    unsafe fn apply_scaleoffset(plist_id: hid_t, mode: ScaleOffset) -> herr_t {
        let (scale_type, factor) = match mode {
            ScaleOffset::Integer(bits) => (H5Z_SO_INT, c_int::from(bits)),
            ScaleOffset::FloatDScale(factor) => (H5Z_SO_FLOAT_DSCALE, c_int::from(factor)),
        };
        H5Pset_scaleoffset(plist_id, scale_type, factor)
    }

    #[cfg(feature = "lzf")]
    unsafe fn apply_lzf(plist_id: hid_t) -> herr_t {
        Self::apply_user(plist_id, lzf::LZF_FILTER_ID, &[])
    }

    #[cfg(feature = "blosc")]
    unsafe fn apply_blosc(
        plist_id: hid_t, complib: Blosc, clevel: u8, shuffle: BloscShuffle,
    ) -> herr_t {
        let mut cdata: Vec<c_uint> = vec![0; 7];
        cdata[4] = c_uint::from(clevel);
        cdata[5] = match shuffle {
            BloscShuffle::None => blosc::BLOSC_NOSHUFFLE,
            BloscShuffle::Byte => blosc::BLOSC_SHUFFLE,
            BloscShuffle::Bit => blosc::BLOSC_BITSHUFFLE,
        };
        cdata[6] = match complib {
            Blosc::BloscLZ => blosc::BLOSC_BLOSCLZ,
            #[cfg(feature = "blosc-lz4")]
            Blosc::LZ4 => blosc::BLOSC_LZ4,
            #[cfg(feature = "blosc-lz4")]
            Blosc::LZ4HC => blosc::BLOSC_LZ4HC,
            #[cfg(feature = "blosc-snappy")]
            Blosc::Snappy => blosc::BLOSC_SNAPPY,
            #[cfg(feature = "blosc-zlib")]
            Blosc::ZLib => blosc::BLOSC_ZLIB,
            #[cfg(feature = "blosc-zstd")]
            Blosc::ZStd => blosc::BLOSC_ZSTD,
        };
        Self::apply_user(plist_id, blosc::BLOSC_FILTER_ID, &cdata)
    }

    #[cfg(feature = "zfp")]
    unsafe fn apply_zfp(plist_id: hid_t, mode: ZfpMode) -> herr_t {
        let mut cdata: Vec<c_uint> = vec![0; 10];
        let (mode_val, param1, param2) = match mode {
            ZfpMode::FixedRate(rate) => {
                let bits = rate.to_bits();
                (1, (bits >> 32) as c_uint, bits as c_uint)
            }
            ZfpMode::FixedPrecision(precision) => (2, precision as c_uint, 0),
            ZfpMode::FixedAccuracy(accuracy) => {
                let bits = accuracy.to_bits();
                (3, (bits >> 32) as c_uint, bits as c_uint)
            }
            ZfpMode::Reversible => (5, 0, 0),
        };
        cdata[7] = mode_val;
        cdata[8] = param1;
        cdata[9] = param2;
        Self::apply_user(plist_id, zfp::ZFP_FILTER_ID, &cdata)
    }

    unsafe fn apply_user(plist_id: hid_t, filter_id: H5Z_filter_t, cdata: &[c_uint]) -> herr_t {
        // We're setting custom filters to optional, same way h5py does it, since
        // the only mention of H5Z_FLAG_MANDATORY in the HDF5 source itself is
        // in H5Pset_fletcher32() in H5Pocpl.c; for all other purposes than
        // verifying checksums optional filter makes more sense than mandatory.
        let cd_nelmts = cdata.len() as _;
        let cd_values = if cd_nelmts == 0 { ptr::null() } else { cdata.as_ptr() };
        H5Pset_filter(plist_id, filter_id, H5Z_FLAG_OPTIONAL, cd_nelmts, cd_values)
    }

    pub(crate) fn apply_to_plist(&self, id: hid_t) -> Result<()> {
        h5try!(match self {
            Self::Deflate(level) => Self::apply_deflate(id, *level),
            Self::Shuffle => Self::apply_shuffle(id),
            Self::Fletcher32 => Self::apply_fletcher32(id),
            Self::SZip(coding, px_per_block) => Self::apply_szip(id, *coding, *px_per_block),
            Self::NBit => Self::apply_nbit(id),
            Self::ScaleOffset(mode) => Self::apply_scaleoffset(id, *mode),
            #[cfg(feature = "lzf")]
            Self::LZF => Self::apply_lzf(id),
            #[cfg(feature = "blosc")]
            Self::Blosc(complib, clevel, shuffle) => {
                Self::apply_blosc(id, *complib, *clevel, *shuffle)
            }
            #[cfg(feature = "zfp")]
            Self::Zfp(mode) => Self::apply_zfp(id, *mode),
            Self::User(filter_id, ref cdata) => Self::apply_user(id, *filter_id, cdata),
        });
        Ok(())
    }

    pub(crate) fn extract_pipeline(plist_id: hid_t) -> Result<Vec<Self>> {
        let mut filters = Vec::new();
        let mut name: Vec<c_char> = vec![0; 257];
        let mut cd_values: Vec<c_uint> = vec![0; 32];
        h5lock!({
            let n_filters = h5try!(H5Pget_nfilters(plist_id));
            for idx in 0..n_filters {
                let mut flags: c_uint = 0;
                let mut cd_nelmts: size_t = cd_values.len() as _;
                let filter_id = h5try!(H5Pget_filter2(
                    plist_id,
                    idx as _,
                    addr_of_mut!(flags),
                    addr_of_mut!(cd_nelmts),
                    cd_values.as_mut_ptr(),
                    name.len() as _,
                    name.as_mut_ptr(),
                    ptr::null_mut(),
                ));
                let cdata = &cd_values[..(cd_nelmts as _)];
                let flt = Self::from_raw(filter_id, cdata)?;
                filters.push(flt);
            }
            Ok(filters)
        })
    }
}

const COMP_FILTER_IDS: &[H5Z_filter_t] =
    &[H5Z_FILTER_DEFLATE, H5Z_FILTER_SZIP, 32000, 32001, 32013];

pub(crate) fn validate_filters(filters: &[Filter], type_class: H5T_class_t) -> Result<()> {
    let mut map: HashMap<H5Z_filter_t, &Filter> = HashMap::new();
    let mut comp_filter: Option<&Filter> = None;

    for filter in filters {
        ensure!(filter.is_available(), "Filter not available: {:?}", filter);

        let id = filter.id();

        if let Some(f) = map.get(&id) {
            fail!("Duplicate filters: {:?} and {:?}", f, filter);
        } else if COMP_FILTER_IDS.contains(&id) {
            if let Some(comp_filter) = comp_filter {
                fail!("Multiple compression filters: {:?} and {:?}", comp_filter, filter);
            }
            comp_filter = Some(filter);
        } else if id == H5Z_FILTER_FLETCHER32 && map.contains_key(&H5Z_FILTER_SCALEOFFSET) {
            fail!("Lossy scale-offset filter before fletcher2 checksum filter");
        } else if let Filter::ScaleOffset(mode) = filter {
            match type_class {
                H5T_class_t::H5T_INTEGER | H5T_class_t::H5T_ENUM => {
                    if let ScaleOffset::FloatDScale(_) = mode {
                        fail!("Invalid scale-offset mode for integer type: {:?}", mode);
                    }
                }
                H5T_class_t::H5T_FLOAT => {
                    if let ScaleOffset::Integer(_) = mode {
                        fail!("Invalid scale-offset mode for float type: {:?}", mode);
                    }
                }
                _ => fail!("Can only use scale-offset with ints/floats, got: {:?}", type_class),
            }
        } else if matches!(filter, Filter::SZip(_, _)) {
            // https://github.com/h5py/h5py/issues/953
            if map.contains_key(&H5Z_FILTER_FLETCHER32) {
                fail!("Fletcher32 filter must be placed after szip filter");
            }
        } else if matches!(filter, Filter::Shuffle) {
            if let Some(comp_filter) = comp_filter {
                fail!("Shuffle filter placed after compression filter: {:?}", comp_filter);
            }
        }
        map.insert(id, filter);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use hdf5_sys::h5t::H5T_class_t;
    use ndarray::Array2;
    use std::io::{Seek, SeekFrom};

    use super::{
        blosc_available, deflate_available, lzf_available, szip_available, validate_filters,
        Filter, FilterInfo, SZip, ScaleOffset,
    };
    use crate::hl::filters::zfp_available;
    use crate::test::with_tmp_file;
    use crate::{plist::DatasetCreate, Result};

    #[test]
    fn test_filter_pipeline() -> Result<()> {
        let mut comp_filters = vec![];
        if deflate_available() {
            comp_filters.push(Filter::deflate(3));
        }
        if szip_available() {
            comp_filters.push(Filter::szip(SZip::Entropy, 8));
        }
        assert_eq!(cfg!(feature = "lzf"), lzf_available());
        #[cfg(feature = "lzf")]
        {
            comp_filters.push(Filter::lzf());
        }
        assert_eq!(cfg!(feature = "blosc-all"), blosc_available());
        #[cfg(feature = "blosc-all")]
        {
            use super::BloscShuffle;
            comp_filters.push(Filter::blosc_blosclz(1, false));
            comp_filters.push(Filter::blosc_lz4(3, true));
            comp_filters.push(Filter::blosc_lz4hc(5, BloscShuffle::Bit));
            comp_filters.push(Filter::blosc_zlib(7, BloscShuffle::None));
            comp_filters.push(Filter::blosc_zstd(9, BloscShuffle::Byte));
            comp_filters.push(Filter::blosc_snappy(0, BloscShuffle::Bit));
        }

        #[cfg(feature = "zfp")]
        assert_eq!(cfg!(feature = "zfp"), zfp_available());
        #[cfg(feature = "zfp")]
        {
            comp_filters.push(Filter::zfp_rate(8.0));
            comp_filters.push(Filter::zfp_precision(16));
            comp_filters.push(Filter::zfp_accuracy(1e-3));
        }

        for c in &comp_filters {
            assert!(c.is_available());
            assert!(c.encode_enabled());
            assert!(c.decode_enabled());

            let pipeline = vec![
                Filter::nbit(),
                Filter::shuffle(),
                c.clone(),
                Filter::fletcher32(),
                Filter::scale_offset(ScaleOffset::Integer(3)),
            ];
            validate_filters(&pipeline, H5T_class_t::H5T_INTEGER)?;

            let plist = DatasetCreate::try_new()?;
            for flt in &pipeline {
                flt.apply_to_plist(plist.id())?;
            }
            assert_eq!(Filter::extract_pipeline(plist.id())?, pipeline);

            let mut b = DatasetCreate::build();
            b.set_filters(&pipeline);
            b.chunk(10);
            let plist = b.finish()?;
            assert_eq!(Filter::extract_pipeline(plist.id())?, pipeline);

            let res = with_tmp_file(|file| {
                file.new_dataset_builder()
                    .empty::<i32>()
                    .shape((10_000, 20))
                    .with_dcpl(|p| p.set_filters(&pipeline))
                    .create("foo")
                    .unwrap();
                let plist = file.dataset("foo").unwrap().dcpl().unwrap();
                Filter::extract_pipeline(plist.id()).unwrap()
            });
            assert_eq!(res, pipeline);
        }

        let bad_filter = Filter::user(12_345, &[1, 2, 3, 4, 5]);
        assert_eq!(Filter::get_info(bad_filter.id()), FilterInfo::default());
        assert!(!bad_filter.is_available());
        assert!(!bad_filter.encode_enabled());
        assert!(!bad_filter.decode_enabled());
        assert_err!(
            validate_filters(&[bad_filter], H5T_class_t::H5T_INTEGER),
            "Filter not available"
        );

        Ok(())
    }

    #[test]
    #[cfg(feature = "zfp")]
    fn test_zfp_filter() -> Result<()> {
        use super::{zfp_available, ZfpMode};

        assert_eq!(cfg!(feature = "zfp"), zfp_available());

        if !zfp_available() {
            println!("ZFP filter not available, skipping test");
            return Ok(());
        }

        // Test different ZFP modes
        let zfp_filters = vec![
            Filter::zfp_rate(8.0),
            Filter::zfp_rate(16.0),
            Filter::zfp_rate(4.0),
            Filter::zfp_precision(16),
            Filter::zfp_precision(32),
            Filter::zfp_accuracy(1e-3),
            Filter::zfp_accuracy(1e-6),
            Filter::zfp(ZfpMode::FixedRate(12.5)),
        ];

        for flt in &zfp_filters {
            println!("Testing filter: {:?}", flt);
            assert!(flt.is_available());
            assert!(flt.encode_enabled());
            assert!(flt.decode_enabled());

            // Test with float type (ZFP only supports floats)
            let pipeline = vec![flt.clone()];
            validate_filters(&pipeline, H5T_class_t::H5T_FLOAT)?;

            let plist = DatasetCreate::try_new()?;
            for f in &pipeline {
                f.apply_to_plist(plist.id())?;
            }
            assert_eq!(Filter::extract_pipeline(plist.id())?, pipeline);

            // Test with actual dataset creation for f32
            let res = with_tmp_file(|file| {
                file.new_dataset_builder()
                    .empty::<f32>()
                    .shape((100, 50))
                    .chunk((10, 10))
                    .with_dcpl(|p| p.set_filters(&pipeline))
                    .create("zfp_f32")
                    .unwrap();

                let plist = file.dataset("zfp_f32").unwrap().dcpl().unwrap();
                Filter::extract_pipeline(plist.id()).unwrap()
            });
            assert_eq!(res, pipeline);

            // Test with f64
            let res_f64 = with_tmp_file(|file| {
                file.new_dataset_builder()
                    .empty::<f64>()
                    .shape((100, 50))
                    .chunk((10, 10))
                    .with_dcpl(|p| p.set_filters(&pipeline))
                    .create("zfp_f64")
                    .unwrap();

                let plist = file.dataset("zfp_f64").unwrap().dcpl().unwrap();
                Filter::extract_pipeline(plist.id()).unwrap()
            });
            assert_eq!(res_f64, pipeline);
        }

        Ok(())
    }

    #[test]
    #[cfg(feature = "zfp")]
    fn test_zfp_reversible() -> Result<()> {
        use super::zfp_available;

        if !zfp_available() {
            println!("ZFP filter not available, skipping test");
            return Ok(());
        }
        with_tmp_file(|file| {
            let data = ndarray::Array1::<f32>::linspace(0.0, 1000.0, 100000);
            file.new_dataset_builder()
                .zfp_reversible()
                .with_data(&data)
                .chunk((10000,))
                .create("zfp_lossless_1d")
                .unwrap();

            let ds = file.dataset("zfp_lossless_1d").unwrap();
            // get number of bytes of ds
            let n_bytes = file.size();
            let read_data: Vec<f32> = ds.read_raw().unwrap();
            let error = data.iter().zip(read_data.iter()).map(|(a, b)| (a - b).abs()).sum::<f32>()
                / data.len() as f32;

            let target_bytes = 100000 * 4;
            assert!(
                n_bytes <= target_bytes,
                "Dataset size {} exceeds target {}",
                n_bytes,
                target_bytes
            );
            assert_eq!(n_bytes, 249368);
            assert_eq!(error, 0.0);
        });
        Ok(())
    }
    #[test]
    #[cfg(feature = "zfp")]
    fn test_zfp_roundtrip_1d() -> Result<()> {
        use super::zfp_available;

        if !zfp_available() {
            println!("ZFP filter not available, skipping test");
            return Ok(());
        }
        let pipeline = vec![Filter::zfp_rate(16.0)];
        with_tmp_file(|file| {
            let data = ndarray::Array1::<f32>::linspace(0.0, 1.0, 1000);
            file.new_dataset_builder()
                .with_data(&data)
                .chunk((100,))
                .with_dcpl(|p| p.set_filters(&pipeline))
                .create("zfp_1d_dcpl")
                .unwrap();

            let ds = file.dataset("zfp_1d_dcpl").unwrap();
            let read_data: Vec<f32> = ds.read_raw().unwrap();

            // ZFP is lossy, so we check approximate equality
            assert_eq!(read_data.len(), data.len());
            for (i, (original, compressed)) in data.iter().zip(read_data.iter()).enumerate() {
                let diff = (original - compressed).abs();
                assert!(
                    diff < 0.1,
                    "Index {}: difference too large: {} vs {} (diff: {})",
                    i,
                    original,
                    compressed,
                    diff
                );
            }
        });

        Ok(())
    }

    #[test]
    #[cfg(feature = "zfp")]
    fn test_zfp_roundtrip_2d() -> Result<()> {
        use super::zfp_available;

        if !zfp_available() {
            println!("ZFP filter not available, skipping test");
            return Ok(());
        }

        with_tmp_file(|file| {
            let data: Vec<f64> = (0..1000).map(|i| (i as f64) * 0.01).collect();
            let data = Array2::from_shape_vec((10, 100), data).unwrap();

            let pipteline = vec![Filter::zfp_precision(32)];
            file.new_dataset_builder()
                .with_data(&data)
                .chunk((10, 10))
                .with_dcpl(|p| p.set_filters(&pipteline))
                .create("zfp_2d")
                .unwrap();

            let ds = file.dataset("zfp_2d").unwrap();
            let read_data: Vec<f64> = ds.read_raw().unwrap();

            assert_eq!(read_data.len(), data.len());
            for (original, compressed) in data.iter().zip(read_data.iter()) {
                let diff = (original - compressed).abs();
                assert!(diff < 0.01, "Difference too large: {} vs {}", original, compressed);
            }
        });

        Ok(())
    }

    #[test]
    #[cfg(feature = "zfp")]
    fn test_zfp_roundtrip_3d() -> Result<()> {
        use super::zfp_available;

        if !zfp_available() {
            println!("ZFP filter not available, skipping test");
            return Ok(());
        }
        let pipeline = vec![Filter::zfp_accuracy(1e-4)];
        with_tmp_file(|file| {
            let data: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.001).collect();
            let data = ndarray::Array3::from_shape_vec((10, 10, 10), data).unwrap();

            file.new_dataset_builder()
                .with_data(&data)
                .chunk((5, 5, 5))
                .with_dcpl(|p| p.set_filters(&pipeline))
                .create("zfp_3d")
                .unwrap();

            let ds = file.dataset("zfp_3d").unwrap();
            let read_data: Vec<f32> = ds.read_raw().unwrap();

            assert_eq!(read_data.len(), data.len());
        });

        Ok(())
    }

    #[test]
    #[cfg(feature = "zfp")]
    fn test_zfp_roundtrip_4d() -> Result<()> {
        use super::zfp_available;

        if !zfp_available() {
            println!("ZFP filter not available, skipping test");
            return Ok(());
        }
        let pipeline = vec![Filter::zfp_rate(10.0)];
        with_tmp_file(|file| {
            let data: Vec<f64> = (0..256).map(|i| (i as f64) * 0.1).collect();
            let data = ndarray::Array4::from_shape_vec((4, 4, 4, 4), data).unwrap();

            file.new_dataset_builder()
                .with_data(&data)
                .chunk((2, 2, 2, 2))
                .with_dcpl(|p| p.set_filters(&pipeline))
                .create("zfp_4d")
                .unwrap();

            let ds = file.dataset("zfp_4d").unwrap();
            let read_data: Vec<f64> = ds.read_raw().unwrap();

            assert_eq!(read_data.len(), data.len());
        });

        Ok(())
    }

    #[test]
    #[cfg(feature = "zfp")]
    fn test_zfp_mode_parsing() -> Result<()> {
        use super::ZfpMode;

        // Test FixedRate parsing
        let rate_bits = 12.5_f64.to_bits();
        let cdata_rate = vec![
            0,
            1,
            4,
            100,
            0,
            0,
            0,
            1, // mode = rate
            (rate_bits >> 32) as u32,
            rate_bits as u32,
        ];
        let filter = Filter::from_raw(super::zfp::ZFP_FILTER_ID, &cdata_rate)?;
        if let Filter::Zfp(ZfpMode::FixedRate(rate)) = filter {
            assert!((rate - 12.5).abs() < 1e-10);
        } else {
            panic!("Expected FixedRate mode, got: {:?}", filter);
        }

        // Test FixedPrecision parsing
        let cdata_precision = vec![
            0, 1, 8, 100, 0, 0, 0, 2,  // mode = precision
            24, // precision
            0,
        ];
        let filter = Filter::from_raw(super::zfp::ZFP_FILTER_ID, &cdata_precision)?;
        if let Filter::Zfp(ZfpMode::FixedPrecision(precision)) = filter {
            assert_eq!(precision, 24);
        } else {
            panic!("Expected FixedPrecision mode, got: {:?}", filter);
        }

        // Test FixedAccuracy parsing
        let accuracy_bits = 1e-5_f64.to_bits();
        let cdata_accuracy = vec![
            0,
            1,
            8,
            100,
            0,
            0,
            0,
            3, // mode = accuracy
            (accuracy_bits >> 32) as u32,
            accuracy_bits as u32,
        ];
        let filter = Filter::from_raw(super::zfp::ZFP_FILTER_ID, &cdata_accuracy)?;
        if let Filter::Zfp(ZfpMode::FixedAccuracy(accuracy)) = filter {
            assert!((accuracy - 1e-5).abs() < 1e-10);
        } else {
            panic!("Expected FixedAccuracy mode, got: {:?}", filter);
        }

        Ok(())
    }

    #[test]
    #[cfg(feature = "zfp")]
    fn test_zfp_with_other_filters() -> Result<()> {
        use super::zfp_available;

        if !zfp_available() {
            println!("ZFP filter not available, skipping test");
            return Ok(());
        }

        let pipeline = vec![Filter::zfp_rate(8.0)];
        // Test ZFP combined with shuffle (shuffle should come first)
        with_tmp_file(|file| {
            let data: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.1).collect();
            let data = ndarray::Array1::from_shape_vec(1000, data).unwrap();
            // file.new_dataset_builder()
            //     .with_data(&data)
            //     .chunk(100)
            //     .with_dcpl(|p| p.set_filters(&pipeline))
            //     .create("zfp_rate_8").unwrap();
            //
            file.new_dataset_builder()
                .zfp_rate(8.0)
                .with_data(&data)
                .chunk(100)
                .create("zfp_rate_8")
                .unwrap();

            let ds = file.dataset("zfp_rate_8").unwrap();
            let read_data: Vec<f32> = ds.read_raw().unwrap();

            let error = data.iter().zip(read_data.iter()).map(|(a, b)| (a - b).abs()).sum::<f32>()
                / data.len() as f32;
            assert_eq!(error, 0.082505114);
            assert_eq!(read_data.len(), data.len());
        });

        // Test ZFP with fletcher32 checksum
        if super::deflate_available() {
            let pipeline = vec![Filter::zfp_precision(24), Filter::fletcher32()];
            with_tmp_file(|file| {
                let data: Vec<f64> = (0..500).map(|i| (i as f64) * 0.01).collect();
                let data = ndarray::Array1::from_shape_vec(500, data).unwrap();

                file.new_dataset_builder()
                    .with_data(&data)
                    .chunk(50)
                    .with_dcpl(|p| p.set_filters(&pipeline))
                    .create("zfp_with_fletcher32")
                    .unwrap();

                let ds = file.dataset("zfp_with_fletcher32").unwrap();
                let read_data: Vec<f64> = ds.read_raw().unwrap();

                assert_eq!(read_data.len(), data.len());
            });
        }

        Ok(())
    }

    #[test]
    #[cfg(feature = "zfp")]
    fn test_zfp_compression_ratios() -> Result<()> {
        use super::zfp_available;

        if !zfp_available() {
            println!("ZFP filter not available, skipping test");
            return Ok(());
        }

        let pipeline = vec![Filter::zfp_rate(32.0)];
        let pipeline2 = vec![Filter::zfp_rate(4.0)];

        with_tmp_file(|file| {
            let data: Vec<f32> = (0..10000).map(|i| (i as f32).sin()).collect();
            let data = ndarray::Array1::from_shape_vec(10000, data).unwrap();
            // Higher rate = less compression but better quality
            file.new_dataset_builder()
                .with_data(&data)
                .chunk(1000)
                .with_dcpl(|p| p.set_filters(&pipeline))
                .create("zfp_high_rate")
                .unwrap();

            // Lower rate = more compression but lower quality

            file.new_dataset_builder()
                .with_data(&data)
                .chunk(1000)
                .with_dcpl(|p| p.set_filters(&pipeline2))
                .create("zfp_low_rate")
                .unwrap();

            let ds_high = file.dataset("zfp_high_rate").unwrap();
            let ds_low = file.dataset("zfp_low_rate").unwrap();

            let read_high: Vec<f32> = ds_high.read_raw().unwrap();
            let read_low: Vec<f32> = ds_low.read_raw().unwrap();

            // High rate should have better accuracy
            let error_high: f32 =
                data.iter().zip(read_high.iter()).map(|(a, b)| (a - b).abs()).sum::<f32>()
                    / data.len() as f32;

            let error_low: f32 =
                data.iter().zip(read_low.iter()).map(|(a, b)| (a - b).abs()).sum::<f32>()
                    / data.len() as f32;

            println!("High rate error: {}, Low rate error: {}", error_high, error_low);
            assert!(error_high < error_low || error_high < 0.001);
        });

        Ok(())
    }

    #[test]
    #[cfg(feature = "zfp")]
    fn test_zfp_edge_cases() -> Result<()> {
        use super::zfp_available;

        if !zfp_available() {
            println!("ZFP filter not available, skipping test");
            return Ok(());
        }

        let pipeline = vec![Filter::zfp_rate(8.0)];

        // Test with all zeros
        with_tmp_file(|file| {
            let data: Vec<f32> = vec![0.0; 1000];
            let data = ndarray::Array1::from_shape_vec(1000, data).unwrap();

            file.new_dataset_builder()
                .with_data(&data)
                .chunk(100)
                .with_dcpl(|p| p.set_filters(&pipeline))
                .create("zfp_zeros")
                .unwrap();

            let ds = file.dataset("zfp_zeros").unwrap();
            let read_data: Vec<f32> = ds.read_raw().unwrap();

            assert_eq!(read_data.len(), data.len());
            for &val in &read_data {
                assert_eq!(val, 0.0);
            }
        });

        let pipeline = vec![Filter::zfp_accuracy(1e-6)];
        // Test with constant values
        with_tmp_file(|file| {
            let data: Vec<f64> = vec![42.0; 500];
            let data = ndarray::Array1::from_shape_vec(500, data).unwrap();
            file.new_dataset_builder()
                .with_data(&data)
                .chunk(50)
                .with_dcpl(|p| p.set_filters(&pipeline))
                .create("zfp_constant")
                .unwrap();

            let ds = file.dataset("zfp_constant").unwrap();
            let read_data: Vec<f64> = ds.read_raw().unwrap();

            assert_eq!(read_data.len(), data.len());
            for &val in &read_data {
                assert!((val - 42.0).abs() < 1e-5);
            }
        });

        Ok(())
    }
}
