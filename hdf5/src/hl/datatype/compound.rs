use std::ffi::CStr;

use hdf5_sys::h5t::{
    H5T_class_t, H5Tcreate, H5Tget_class, H5Tget_member_name, H5Tget_member_offset,
    H5Tget_member_type, H5Tget_nmembers, H5Tinsert,
};

use super::Datatype;
use crate::internal_prelude::*;

/// A named member of a compound datatype.
///
/// Used to construct compound datatypes with [`Datatype::compound`] and
/// [`Datatype::compound_with_size`], and returned by
/// [`Datatype::compound_members`]. The member datatype keeps its full
/// representation, including properties that
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
    /// Creates a compound datatype with the specified members.
    ///
    /// The size of the compound datatype is the end of its furthest member, so
    /// it has no trailing padding. Use [`compound_with_size`](Self::compound_with_size)
    /// to match a memory layout with trailing padding, such as a `#[repr(C)]` struct.
    ///
    /// Each member datatype is copied, so the members are left unchanged and do
    /// not need to outlive the compound datatype.
    ///
    /// # Errors
    ///
    /// Returns an error if `members` is empty, or if a member is rejected for any
    /// reason listed for [`compound_with_size`](Self::compound_with_size).
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_metno::{Datatype, datatype::{ByteOrder, CompoundMember}};
    ///
    /// let id = Datatype::from_type::<u16>()?.with_byte_order(ByteOrder::BigEndian)?;
    /// let members = [
    ///     CompoundMember {
    ///         name: "value".into(),
    ///         offset: 0,
    ///         datatype: Datatype::from_type::<f64>()?,
    ///     },
    ///     CompoundMember {
    ///         name: "id".into(),
    ///         offset: 8,
    ///         datatype: id,
    ///     },
    /// ];
    ///
    /// let record = Datatype::compound(&members)?;
    /// assert_eq!(record.size(), 10);
    /// assert_eq!(record.compound_members()?, members);
    /// # Ok::<(), hdf5_metno::Error>(())
    /// ```
    pub fn compound(members: &[CompoundMember]) -> Result<Self> {
        let mut size = 0_usize;
        for member in members {
            let end = member.offset.checked_add(member.datatype.size());
            size = size.max(end.ok_or_else(|| {
                format!("Compound member {:?} extends past the maximum size", member.name)
            })?);
        }
        Self::compound_with_size(size, members)
    }

    /// Creates a compound datatype with the specified size in bytes and members.
    ///
    /// Bytes not covered by a member are padding. Each member datatype is copied,
    /// so the members are left unchanged and do not need to outlive the compound
    /// datatype.
    ///
    /// # Errors
    ///
    /// Returns an error if `size` is zero, a member name is empty, not unique, or
    /// contains a null byte, a member extends past `size`, two members overlap,
    /// or HDF5 cannot create the datatype.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::mem::offset_of;
    ///
    /// use hdf5_metno::{Datatype, datatype::CompoundMember};
    ///
    /// #[repr(C)]
    /// struct Record {
    ///     value: f64,
    ///     id: u16,
    /// }
    ///
    /// let members = [
    ///     CompoundMember {
    ///         name: "value".into(),
    ///         offset: offset_of!(Record, value),
    ///         datatype: Datatype::from_type::<f64>()?,
    ///     },
    ///     CompoundMember {
    ///         name: "id".into(),
    ///         offset: offset_of!(Record, id),
    ///         datatype: Datatype::from_type::<u16>()?,
    ///     },
    /// ];
    ///
    /// let record = Datatype::compound_with_size(size_of::<Record>(), &members)?;
    /// assert_eq!(record.size(), size_of::<Record>());
    /// # Ok::<(), hdf5_metno::Error>(())
    /// ```
    pub fn compound_with_size(size: usize, members: &[CompoundMember]) -> Result<Self> {
        ensure!(size > 0, "Compound datatype size must be positive");
        h5lock!({
            let compound = Self::from_id(h5try!(H5Tcreate(H5T_class_t::H5T_COMPOUND, size)))?;
            for member in members {
                let end = member.offset.checked_add(member.datatype.size());
                ensure!(
                    end.is_some_and(|end| end <= size),
                    "Compound member {:?} extends past the compound size",
                    member.name
                );
                let name = to_cstring(member.name.as_str())?;
                h5try!(H5Tinsert(
                    compound.id(),
                    name.as_ptr(),
                    member.offset,
                    member.datatype.id()
                ));
            }
            Ok(compound)
        })
    }

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
