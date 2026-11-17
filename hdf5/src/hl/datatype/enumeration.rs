use std::cell::Cell;
use std::ffi::{CStr, c_void};
use std::ptr;

use hdf5_sys::h5p::{H5Pcreate, H5Pset_type_conv_cb};
use hdf5_sys::h5t::{
    H5T_class_t, H5T_conv_except_t, H5T_conv_ret_t, H5Tconvert, H5Tenum_create, H5Tenum_insert,
    H5Tget_class, H5Tget_member_name, H5Tget_member_value, H5Tget_nmembers,
};

use super::{ByteOrder, Datatype, IntegerLayout};
use crate::globals::H5P_DATASET_XFER;
use crate::hl::plist::PropertyList;
use crate::internal_prelude::*;

/// A named value of an enum datatype.
///
/// Used to construct enum datatypes with [`Datatype::enumeration`], and returned
/// by [`Datatype::enum_variants`]. The value is encoded to and decoded from the
/// representation of the enum's integer base datatype, including its byte order,
/// bit offset, precision, and sign.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnumVariant {
    /// The variant name, unique within the enum datatype.
    pub name: String,
    /// The variant value.
    pub value: i128,
}

/// A named value of an enum datatype, as stored.
///
/// Returned by [`Datatype::encoded_enum_variants`]. The value holds the bytes
/// stored for the variant, in the representation of the enum's integer base
/// datatype, including its padding bits. Unlike [`EnumVariant`], it can
/// represent values of any width.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedEnumVariant {
    /// The variant name, unique within the enum datatype.
    pub name: String,
    /// The stored value, with the same size in bytes as the enum datatype.
    pub value: Box<[u8]>,
}

impl Datatype {
    /// Creates an enum datatype with the specified integer base datatype and
    /// variants.
    ///
    /// Each value is encoded according to the base datatype, including its byte
    /// order, bit offset, precision, sign, and padding. The base is copied, so it
    /// is left unchanged and does not need to outlive the enum datatype.
    ///
    /// # Errors
    ///
    /// Returns an error if `base` is not an integer datatype, a value is out of
    /// range for the base, a variant name is empty, not unique, or contains a null
    /// byte, two variants have the same value, or HDF5 cannot create the datatype.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_metno::{Datatype, datatype::{ByteOrder, EncodedEnumVariant, EnumVariant}};
    ///
    /// let base = Datatype::from_type::<i16>()?.with_byte_order(ByteOrder::BigEndian)?;
    /// let variants = [
    ///     EnumVariant { name: "Low".into(), value: -2 },
    ///     EnumVariant { name: "High".into(), value: 258 },
    /// ];
    ///
    /// let level = Datatype::enumeration(&base, &variants)?;
    /// assert_eq!(level.enum_variants()?, variants);
    /// assert_eq!(
    ///     level.encoded_enum_variants()?[1],
    ///     EncodedEnumVariant { name: "High".into(), value: [0x01, 0x02].into() }
    /// );
    /// # Ok::<(), hdf5_metno::Error>(())
    /// ```
    pub fn enumeration(base: &Self, variants: &[EnumVariant]) -> Result<Self> {
        h5lock!({
            let enum_type = Self::from_id(h5try!(H5Tenum_create(base.id())))?;
            let codec = EnumValueCodec::new(base)?;
            for variant in variants {
                let name = to_cstring(variant.name.as_str())?;
                let value = codec.encode(variant.value)?.ok_or_else(|| {
                    format!(
                        "Enum variant {:?} value {} is out of range for the base datatype",
                        variant.name, variant.value
                    )
                })?;
                h5try!(H5Tenum_insert(enum_type.id(), name.as_ptr(), value.as_ptr().cast()));
            }
            Ok(enum_type)
        })
    }

    /// Returns the variants of this enum datatype in index order.
    ///
    /// Each value is decoded according to the integer base datatype, so values
    /// stored with a nonnative byte order, bit offset, or precision are reported
    /// correctly. Values must fit in an `i128`.
    ///
    /// # Errors
    ///
    /// Returns an error if this is not an enum datatype, a variant name is not
    /// valid UTF-8, a value does not fit in an `i128`, or HDF5 cannot read a
    /// variant.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_metno::{Datatype, datatype::EnumVariant};
    /// use hdf5_metno::types::{EnumMember, EnumType, IntSize, TypeDescriptor};
    ///
    /// let level = Datatype::from_descriptor(&TypeDescriptor::Enum(EnumType {
    ///     size: IntSize::U1,
    ///     signed: true,
    ///     members: vec![
    ///         EnumMember { name: "Low".into(), value: -1_i64 as u64 },
    ///         EnumMember { name: "High".into(), value: 1 },
    ///     ],
    /// }))?;
    ///
    /// assert_eq!(
    ///     level.enum_variants()?,
    ///     [
    ///         EnumVariant { name: "Low".into(), value: -1 },
    ///         EnumVariant { name: "High".into(), value: 1 },
    ///     ]
    /// );
    /// # Ok::<(), hdf5_metno::Error>(())
    /// ```
    pub fn enum_variants(&self) -> Result<Vec<EnumVariant>> {
        h5lock!({
            let encoded = self.encoded_enum_variants()?;
            let codec = EnumValueCodec::new(&self.base()?)?;
            encoded
                .into_iter()
                .map(|EncodedEnumVariant { name, value }| {
                    let value = codec.decode(&value)?.ok_or_else(|| {
                        format!("Enum member {name:?} value does not fit in i128")
                    })?;
                    Ok(EnumVariant { name, value })
                })
                .collect()
        })
    }

    /// Returns the variants of this enum datatype in index order, with values
    /// as stored.
    ///
    /// Each value is returned without conversion, in the byte representation of
    /// the integer base datatype. Use [`enum_variants`](Self::enum_variants) to
    /// decode the values to integers.
    ///
    /// # Errors
    ///
    /// Returns an error if this is not an enum datatype, a variant name is not
    /// valid UTF-8, or HDF5 cannot read a variant.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_metno::{Datatype, datatype::EncodedEnumVariant};
    /// use hdf5_metno::types::{EnumMember, EnumType, IntSize, TypeDescriptor};
    ///
    /// let level = Datatype::from_descriptor(&TypeDescriptor::Enum(EnumType {
    ///     size: IntSize::U2,
    ///     signed: false,
    ///     members: vec![EnumMember { name: "High".into(), value: 258 }],
    /// }))?;
    ///
    /// assert_eq!(
    ///     level.encoded_enum_variants()?,
    ///     [EncodedEnumVariant { name: "High".into(), value: 258_u16.to_ne_bytes().into() }]
    /// );
    /// # Ok::<(), hdf5_metno::Error>(())
    /// ```
    pub fn encoded_enum_variants(&self) -> Result<Vec<EncodedEnumVariant>> {
        h5lock!({
            ensure!(
                h5try!(H5Tget_class(self.id()) as i32) == H5T_class_t::H5T_ENUM as i32,
                "Expected an enum datatype"
            );
            let size = self.size();
            let count = h5try!(H5Tget_nmembers(self.id()));
            (0..count as u32)
                .map(|index| {
                    let name_ptr = H5Tget_member_name(self.id(), index);
                    if name_ptr.is_null() {
                        return Err(Error::query().unwrap_or_else(|error| error));
                    }
                    let name = CStr::from_ptr(name_ptr).to_str().map(str::to_owned);
                    h5_free_memory(name_ptr.cast());
                    let name = name.map_err(|_| "Enum member name is not valid UTF-8")?;
                    let mut value = vec![0_u8; size];
                    h5try!(H5Tget_member_value(self.id(), index, value.as_mut_ptr().cast()));
                    Ok(EncodedEnumVariant { name, value: value.into_boxed_slice() })
                })
                .collect()
        })
    }
}

/// Converts enum values between an integer base datatype and `i128`.
struct EnumValueCodec {
    base: Datatype,
    native: Datatype,
    transfer: PropertyList,
    out_of_range: Box<Cell<bool>>,
}

impl EnumValueCodec {
    fn new(base: &Datatype) -> Result<Self> {
        extern "C" fn abort_out_of_range(
            except_type: H5T_conv_except_t, _src_id: hid_t, _dst_id: hid_t, _src_buf: *mut c_void,
            _dst_buf: *mut c_void, out_of_range: *mut c_void,
        ) -> H5T_conv_ret_t {
            match except_type {
                H5T_conv_except_t::H5T_CONV_EXCEPT_RANGE_HI
                | H5T_conv_except_t::H5T_CONV_EXCEPT_RANGE_LOW => {
                    unsafe { (*out_of_range.cast::<Cell<bool>>()).set(true) };
                    H5T_conv_ret_t::H5T_CONV_ABORT
                }
                _ => H5T_conv_ret_t::H5T_CONV_UNHANDLED,
            }
        }

        let native = Datatype::from_integer_layout(
            &IntegerLayout::builder()
                .size(size_of::<i128>() as u32)
                .byte_order(if cfg!(target_endian = "little") {
                    ByteOrder::LittleEndian
                } else {
                    ByteOrder::BigEndian
                })
                .signed(true)
                .bit_precision(i128::BITS as u16)
                .build()?,
        )?;
        let out_of_range = Box::new(Cell::new(false));
        h5lock!({
            let transfer = PropertyList::from_id(h5try!(H5Pcreate(*H5P_DATASET_XFER)))?;
            h5try!(H5Pset_type_conv_cb(
                transfer.id(),
                Some(abort_out_of_range),
                ptr::from_ref(out_of_range.as_ref()).cast_mut().cast()
            ));
            Ok(Self { base: base.clone(), native, transfer, out_of_range })
        })
    }

    /// Decodes a value in the base representation, or `None` if it does not fit in `i128`.
    fn decode(&self, raw: &[u8]) -> Result<Option<i128>> {
        let mut buffer = vec![0_u8; raw.len().max(size_of::<i128>())];
        buffer[..raw.len()].copy_from_slice(raw);
        Ok(self
            .convert(&self.base, &self.native, &mut buffer)?
            .then(|| i128::from_ne_bytes(buffer[..size_of::<i128>()].try_into().unwrap())))
    }

    /// Encodes a value in the base representation, or `None` if it does not fit in the base.
    fn encode(&self, value: i128) -> Result<Option<Vec<u8>>> {
        let mut buffer = vec![0_u8; self.base.size().max(size_of::<i128>())];
        buffer[..size_of::<i128>()].copy_from_slice(&value.to_ne_bytes());
        Ok(self.convert(&self.native, &self.base, &mut buffer)?.then(|| {
            buffer.truncate(self.base.size());
            buffer
        }))
    }

    fn convert(&self, src: &Datatype, dst: &Datatype, buffer: &mut [u8]) -> Result<bool> {
        self.out_of_range.set(false);
        let converted = h5call!(H5Tconvert(
            src.id(),
            dst.id(),
            1,
            buffer.as_mut_ptr().cast(),
            ptr::null_mut(),
            self.transfer.id()
        ));
        if self.out_of_range.get() {
            return Ok(false);
        }
        converted?;
        Ok(true)
    }
}
