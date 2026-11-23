use std::cell::Cell;
use std::ffi::{CStr, c_void};
use std::ptr;

use hdf5_sys::h5p::{H5Pcreate, H5Pset_type_conv_cb};
use hdf5_sys::h5t::{
    H5T_class_t, H5T_conv_except_t, H5T_conv_ret_t, H5Tconvert, H5Tget_class, H5Tget_member_name,
    H5Tget_member_value, H5Tget_nmembers, H5Tget_super,
};

use super::{ByteOrder, Datatype, IntegerLayout};
use crate::globals::H5P_DATASET_XFER;
use crate::hl::plist::PropertyList;
use crate::internal_prelude::*;

/// A named value of an enum datatype.
///
/// Returned by [`Datatype::enum_variants`]. The value is decoded from the
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

        h5lock!({
            let encoded = self.encoded_enum_variants()?;
            let base = Self::from_id(h5try!(H5Tget_super(self.id())))?;
            let native = Self::from_integer_layout(
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
            let out_of_range = Cell::new(false);
            let transfer = PropertyList::from_id(h5try!(H5Pcreate(*H5P_DATASET_XFER)))?;
            h5try!(H5Pset_type_conv_cb(
                transfer.id(),
                Some(abort_out_of_range),
                ptr::from_ref(&out_of_range).cast_mut().cast()
            ));
            encoded
                .into_iter()
                .map(|EncodedEnumVariant { name, value }| {
                    let mut buffer = value.into_vec();
                    buffer.resize(buffer.len().max(size_of::<i128>()), 0);
                    let converted = h5call!(H5Tconvert(
                        base.id(),
                        native.id(),
                        1,
                        buffer.as_mut_ptr().cast(),
                        ptr::null_mut(),
                        transfer.id()
                    ));
                    ensure!(
                        !out_of_range.get(),
                        "Enum member {:?} value does not fit in i128",
                        name
                    );
                    converted?;
                    let value =
                        i128::from_ne_bytes(buffer[..size_of::<i128>()].try_into().unwrap());
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
