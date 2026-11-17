use std::ffi::CStr;

use hdf5_sys::h5t::{
    H5T_class_t, H5Tget_class, H5Tget_member_name, H5Tget_member_offset, H5Tget_member_type,
    H5Tget_nmembers,
};

use super::Datatype;
use crate::internal_prelude::*;

/// A named member of a compound datatype.
///
/// Returned by [`Datatype::compound_members`]. The member datatype keeps its
/// full representation, including properties that
/// [`to_descriptor`](Datatype::to_descriptor) cannot represent.
#[derive(Clone, Debug, PartialEq)]
pub struct CompoundMember {
    /// The member name, unique within the compound datatype.
    pub name: String,
    /// The byte offset of the member from the start of the compound datatype.
    pub offset: usize,
    /// The datatype of the member.
    pub datatype: Datatype,
}

impl Datatype {
    /// Returns the members of this compound datatype in index order.
    ///
    /// Each member datatype is an independent copy, so this datatype is left
    /// unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error if this is not a compound datatype, a member name is not
    /// valid UTF-8, or HDF5 cannot read a member.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_metno::Datatype;
    /// use hdf5_metno::types::{CompoundField, CompoundType, FloatSize, IntSize, TypeDescriptor};
    ///
    /// let point = Datatype::from_descriptor(&TypeDescriptor::Compound(CompoundType {
    ///     fields: vec![
    ///         CompoundField::typed::<u16>("id", 0, 0),
    ///         CompoundField::typed::<f64>("value", 8, 1),
    ///     ],
    ///     size: 16,
    /// }))?;
    ///
    /// let members = point.compound_members()?;
    ///
    /// assert_eq!(members[0].name, "id");
    /// assert_eq!(members[0].offset, 0);
    /// assert_eq!(members[0].datatype.to_descriptor()?, TypeDescriptor::Unsigned(IntSize::U2));
    /// assert_eq!(members[1].name, "value");
    /// assert_eq!(members[1].offset, 8);
    /// assert_eq!(members[1].datatype.to_descriptor()?, TypeDescriptor::Float(FloatSize::U8));
    /// # Ok::<(), hdf5_metno::Error>(())
    /// ```
    pub fn compound_members(&self) -> Result<Vec<CompoundMember>> {
        h5lock!({
            ensure!(
                h5try!(H5Tget_class(self.id()) as i32) == H5T_class_t::H5T_COMPOUND as i32,
                "Expected a compound datatype"
            );
            let count = h5try!(H5Tget_nmembers(self.id()));
            (0..count as u32)
                .map(|index| {
                    let name_ptr = H5Tget_member_name(self.id(), index);
                    if name_ptr.is_null() {
                        return Err(Error::query().unwrap_or_else(|error| error));
                    }
                    let name = CStr::from_ptr(name_ptr).to_str().map(str::to_owned);
                    h5_free_memory(name_ptr.cast());
                    let name = name.map_err(|_| "Compound member name is not valid UTF-8")?;
                    Ok(CompoundMember {
                        name,
                        offset: H5Tget_member_offset(self.id(), index),
                        datatype: Self::from_id(h5try!(H5Tget_member_type(self.id(), index)))?,
                    })
                })
                .collect()
        })
    }
}
