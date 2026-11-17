use hdf5_sys::h5t::{H5T_class_t, H5T_str_t, H5Tcopy, H5Tget_class, H5Tget_strpad, H5Tset_strpad};

use super::Datatype;
use crate::internal_prelude::*;

/// The storage of string characters shorter than a string datatype's size.
///
/// The padding also determines how HDF5 converts between strings of different
/// lengths. A null-terminated string is truncated with a terminator appended,
/// while padded strings are truncated without one.
///
/// Inspect the padding of a datatype with [`Datatype::string_padding`], or
/// configure it with [`Datatype::with_string_padding`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StringPadding {
    /// Terminates the string with a null byte, as in C.
    NullTerminated,
    /// Pads the string with null bytes.
    NullPadded,
    /// Pads the string with spaces, as in Fortran.
    SpacePadded,
}

impl From<StringPadding> for H5T_str_t {
    fn from(padding: StringPadding) -> Self {
        match padding {
            StringPadding::NullTerminated => Self::H5T_STR_NULLTERM,
            StringPadding::NullPadded => Self::H5T_STR_NULLPAD,
            StringPadding::SpacePadded => Self::H5T_STR_SPACEPAD,
        }
    }
}

impl TryFrom<H5T_str_t> for StringPadding {
    type Error = Error;

    fn try_from(padding: H5T_str_t) -> Result<Self> {
        match padding {
            H5T_str_t::H5T_STR_NULLTERM => Ok(Self::NullTerminated),
            H5T_str_t::H5T_STR_NULLPAD => Ok(Self::NullPadded),
            H5T_str_t::H5T_STR_SPACEPAD => Ok(Self::SpacePadded),
            _ => fail!("Invalid string padding"),
        }
    }
}

impl Datatype {
    /// Returns the padding of this string datatype.
    ///
    /// Supports both fixed-length and variable-length string datatypes.
    ///
    /// # Errors
    ///
    /// Returns an error if this is not a string datatype or HDF5 cannot read
    /// the padding.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_metno::{Datatype, datatype::StringPadding, types::VarLenUnicode};
    ///
    /// let datatype = Datatype::from_type::<VarLenUnicode>()?;
    /// assert_eq!(datatype.string_padding()?, StringPadding::NullTerminated);
    /// # Ok::<(), hdf5_metno::Error>(())
    /// ```
    pub fn string_padding(&self) -> Result<StringPadding> {
        h5lock!({
            ensure!(
                h5try!(H5Tget_class(self.id()) as i32) == H5T_class_t::H5T_STRING as i32,
                "Expected a string datatype"
            );
            let padding = H5Tget_strpad(self.id());
            h5try!(padding as i32);
            padding.try_into()
        })
    }

    /// Returns a copy of this string datatype with the specified padding.
    ///
    /// The copy is transient even when this datatype is committed, and this
    /// datatype is left unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error if this is not a string datatype, it cannot be copied,
    /// or HDF5 cannot set the padding.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_metno::{Datatype, datatype::StringPadding, types::FixedAscii};
    ///
    /// let original = Datatype::from_type::<FixedAscii<8>>()?;
    /// let space_padded = original.with_string_padding(StringPadding::SpacePadded)?;
    ///
    /// assert_eq!(space_padded.string_padding()?, StringPadding::SpacePadded);
    /// assert_eq!(original.string_padding()?, StringPadding::NullPadded);
    /// # Ok::<(), hdf5_metno::Error>(())
    /// ```
    pub fn with_string_padding(&self, padding: StringPadding) -> Result<Self> {
        h5lock!({
            ensure!(
                h5try!(H5Tget_class(self.id()) as i32) == H5T_class_t::H5T_STRING as i32,
                "Expected a string datatype"
            );
            let datatype = Self::from_id(h5try!(H5Tcopy(self.id())))?;
            h5try!(H5Tset_strpad(datatype.id(), padding.into()));
            Ok(datatype)
        })
    }
}
