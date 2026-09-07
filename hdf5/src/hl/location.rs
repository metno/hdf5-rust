use std::fmt::{self, Debug};
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::ptr;

use hdf5_sys::h5o::H5Ocopy;
#[allow(deprecated)]
use hdf5_sys::h5o::H5Oset_comment;
#[cfg(feature = "1.10.3")]
use hdf5_sys::h5o::{H5O_INFO_BASIC, H5O_INFO_NUM_ATTRS, H5O_INFO_TIME};
#[cfg(feature = "1.12.0")]
use hdf5_sys::h5o::{
    H5O_info2_t, H5O_token_t, H5Oget_info_by_name3, H5Oget_info3, H5Oopen_by_token,
};
#[cfg(not(feature = "1.10.3"))]
use hdf5_sys::h5o::{H5Oget_info_by_name1, H5Oget_info1};
#[cfg(all(feature = "1.10.3", not(feature = "1.12.0")))]
use hdf5_sys::h5o::{H5Oget_info_by_name2, H5Oget_info2};
#[cfg(not(feature = "1.12.0"))]
use hdf5_sys::{h5::haddr_t, h5o::H5O_info1_t, h5o::H5Oopen_by_addr};
use hdf5_sys::{
    h5a::{H5A_info_t, H5Adelete, H5Aget_info_by_name, H5Aiterate2, H5Aopen, H5Aopen_by_idx},
    h5f::H5Fget_name,
    h5i::{H5Iget_file_id, H5Iget_name},
    h5o::{H5O_type_t, H5Oget_comment},
};

use crate::internal_prelude::*;

use super::attribute::{AttrInfo, AttributeBuilderEmpty};
use super::iteration::visit;

/// Named location (file, group, dataset, named datatype).
#[repr(transparent)]
#[derive(Clone)]
pub struct Location(Handle);

impl ObjectClass for Location {
    const NAME: &'static str = "location";
    const VALID_TYPES: &'static [H5I_type_t] =
        &[H5I_FILE, H5I_GROUP, H5I_DATATYPE, H5I_DATASET, H5I_ATTR];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }

    fn short_repr(&self) -> Option<String> {
        Some(format!("\"{}\"", self.name()))
    }
}

impl Debug for Location {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.debug_fmt(f)
    }
}

impl Deref for Location {
    type Target = Object;

    fn deref(&self) -> &Object {
        unsafe { self.transmute() }
    }
}

impl Location {
    /// Returns the name of the object within the file, or empty string if the object doesn't
    /// have a name (e.g., an anonymous dataset).
    pub fn name(&self) -> String {
        // TODO: should this return Result<String> or an empty string if it fails?
        h5lock!(get_h5_str(|m, s| H5Iget_name(self.id(), m, s)).unwrap_or_else(|_| String::new()))
    }

    /// Returns the name of the file containing the named object (or the file itself).
    pub fn filename(&self) -> String {
        // TODO: should this return Result<String> or an empty string if it fails?
        h5lock!(get_h5_str(|m, s| H5Fget_name(self.id(), m, s)).unwrap_or_else(|_| String::new()))
    }

    /// Returns a handle to the file containing the named object (or the file itself).
    pub fn file(&self) -> Result<File> {
        File::from_id(h5try!(H5Iget_file_id(self.id())))
    }

    /// Returns the comment attached to the named object, if any.
    pub fn comment(&self) -> Option<String> {
        // TODO: should this return Result<Option<String>> or fail silently?
        let comment = h5lock!(get_h5_str(|m, s| H5Oget_comment(self.id(), m, s)).ok());
        comment.and_then(|c| if c.is_empty() { None } else { Some(c) })
    }

    /// Set the comment attached to the named object.
    #[deprecated(note = "attributes are preferred to comments")]
    pub fn set_comment(&self, comment: &str) -> Result<()> {
        // TODO: &mut self?
        let comment = to_cstring(comment)?;
        #[allow(deprecated)]
        h5call!(H5Oset_comment(self.id(), comment.as_ptr())).and(Ok(()))
    }

    /// Clear the comment attached to the named object.
    #[deprecated(note = "attributes are preferred to comments")]
    pub fn clear_comment(&self) -> Result<()> {
        // TODO: &mut self?
        #[allow(deprecated)]
        h5call!(H5Oset_comment(self.id(), ptr::null_mut())).and(Ok(()))
    }

    /// Create a builder for a new attribute of known type.
    pub fn new_attr<T: H5Type>(&self) -> AttributeBuilderEmpty {
        AttributeBuilder::new(self).empty::<T>()
    }

    /// Create a builder for a new attribute.
    pub fn new_attr_builder(&self) -> AttributeBuilder {
        AttributeBuilder::new(self)
    }

    /// Create a new named attribute on the object.
    pub fn attr(&self, name: &str) -> Result<Attribute> {
        let name = to_cstring(name)?;
        Attribute::from_id(h5try!(H5Aopen(self.id(), name.as_ptr(), H5P_DEFAULT)))
    }

    /// Opens the attribute at `index` along `index_type` in `iteration_order`.
    ///
    /// # Errors
    ///
    /// Fails if `index` is not below the number of attributes.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_metno::{File, IndexType, IterationOrder};
    ///
    /// let file = File::with_options().with_fapl(|p| p.core_filebacked(false)).create("attr_by_index.h5")?;
    /// file.new_attr::<u32>().create("b")?;
    /// file.new_attr::<u32>().create("a")?;
    ///
    /// let last = file.attr_by_index(IndexType::Name, IterationOrder::Decreasing, 0)?;
    /// assert_eq!(last.name(), "b");
    /// # Ok::<(), hdf5_metno::Error>(())
    /// ```
    pub fn attr_by_index(
        &self, index_type: IndexType, iteration_order: IterationOrder, index: u64,
    ) -> Result<Attribute> {
        Attribute::from_id(h5try!(H5Aopen_by_idx(
            self.id(),
            b".\0".as_ptr().cast::<c_char>(),
            index_type.into(),
            iteration_order.into(),
            index,
            H5P_DEFAULT,
            H5P_DEFAULT
        )))
    }

    /// Returns information about the attribute called `name`.
    ///
    /// # Errors
    ///
    /// Fails if the object has no attribute called `name`.
    pub fn attr_info(&self, name: &str) -> Result<AttrInfo> {
        let name = to_cstring(name)?;
        let mut info = MaybeUninit::<H5A_info_t>::uninit();
        h5call!(H5Aget_info_by_name(
            self.id(),
            b".\0".as_ptr().cast::<c_char>(),
            name.as_ptr(),
            info.as_mut_ptr(),
            H5P_DEFAULT
        ))?;
        // SAFETY: H5Aget_info_by_name fills the info on success, and the error was checked
        let info = unsafe { info.assume_init() };
        Ok(AttrInfo::from(&info))
    }

    /// Returns the names of all attributes of the object, in increasing name order.
    pub fn attr_names(&self) -> Result<Vec<String>> {
        self.attr_names_by(IndexType::Name, IterationOrder::Increasing)
    }

    /// Returns the names of all attributes of the object along `index_type` in
    /// `iteration_order`.
    pub fn attr_names_by(
        &self, index_type: IndexType, iteration_order: IterationOrder,
    ) -> Result<Vec<String>> {
        let mut names = vec![];
        self.iter_attrs(index_type, iteration_order, |name, _| {
            names.push(name.to_owned());
            Ok(())
        })?;
        Ok(names)
    }

    /// Returns the name and [`AttrInfo`] of all attributes of the object along `index_type`
    /// in `iteration_order`.
    pub fn attrs(
        &self, index_type: IndexType, iteration_order: IterationOrder,
    ) -> Result<Vec<(String, AttrInfo)>> {
        let mut attrs = vec![];
        self.iter_attrs(index_type, iteration_order, |name, info| {
            attrs.push((name.to_owned(), info));
            Ok(())
        })?;
        Ok(attrs)
    }

    /// Visits every attribute of the object.
    ///
    /// The attributes are traversed along `index_type` in `iteration_order`, and `op`
    /// is called with the name and the [`AttrInfo`] of each attribute. Use
    /// [`find_attr`](Self::find_attr) to stop early.
    ///
    /// An object that does not track attribute creation order is still traversed by
    /// [`IndexType::CreationOrder`]. Attributes stored compactly in the object header
    /// come in storage sequence with a position each. Attributes stored densely come
    /// from the name index in native order with no positions. Track the order with
    /// [`AttrCreationOrder`](crate::plist::group_create::AttrCreationOrder) for a
    /// reliable result.
    ///
    /// # Errors
    ///
    /// Returns the first error returned by `op`, or the HDF5 error if the iteration
    /// itself fails.
    ///
    /// # Panics
    ///
    /// A panic in `op` is caught while HDF5 frames are on the stack and resumed once
    /// the iteration has returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_metno::plist::group_create::AttrCreationOrder;
    /// use hdf5_metno::{File, IndexType, IterationOrder};
    ///
    /// let file = File::with_options().with_fapl(|p| p.core_filebacked(false)).create("iter_attrs.h5")?;
    /// let group = file
    ///     .create_group_builder()
    ///     .with_gcpl(|gcpl| gcpl.attr_creation_order(AttrCreationOrder::Tracked))
    ///     .create("g")?;
    /// group.new_attr::<u32>().create("b")?;
    /// group.new_attr::<u32>().create("a")?;
    ///
    /// let mut names = vec![];
    /// group.iter_attrs(IndexType::CreationOrder, IterationOrder::Increasing, |name, _| {
    ///     names.push(name.to_owned());
    ///     Ok(())
    /// })?;
    /// assert_eq!(names, ["b", "a"]);
    /// assert_eq!(group.attr_names()?, ["a", "b"]);
    /// # Ok::<(), hdf5_metno::Error>(())
    /// ```
    pub fn iter_attrs<F>(
        &self, index_type: IndexType, iteration_order: IterationOrder, mut op: F,
    ) -> Result<()>
    where
        F: FnMut(&str, AttrInfo) -> Result<()>,
    {
        self.iter_attrs_from(IterationCursor::start(index_type, iteration_order), |name, info| {
            op(name, info)?;
            Ok(None::<()>)
        })?;
        Ok(())
    }

    /// Visits the attributes of the object until `op` returns a value.
    ///
    /// The attributes are traversed along `index_type` in `iteration_order`, and `op`
    /// is called with the name and the [`AttrInfo`] of each attribute until it returns
    /// `Some`. That value is returned, or `None` once every attribute was visited.
    ///
    /// # Errors
    ///
    /// As for [`iter_attrs`](Self::iter_attrs).
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_metno::{File, IndexType, IterationOrder};
    ///
    /// let file = File::with_options().with_fapl(|p| p.core_filebacked(false)).create("find_attr.h5")?;
    /// file.new_attr::<u8>().create("small")?;
    /// file.new_attr::<u64>().create("large")?;
    ///
    /// let wide = file.find_attr(IndexType::Name, IterationOrder::Increasing, |name, info| {
    ///     if info.data_size > 4 { Ok(Some(name.to_owned())) } else { Ok(None) }
    /// })?;
    /// assert_eq!(wide, Some("large".to_owned()));
    /// # Ok::<(), hdf5_metno::Error>(())
    /// ```
    pub fn find_attr<B, F>(
        &self, index_type: IndexType, iteration_order: IterationOrder, op: F,
    ) -> Result<Option<B>>
    where
        F: FnMut(&str, AttrInfo) -> Result<Option<B>>,
    {
        match self.iter_attrs_from(IterationCursor::start(index_type, iteration_order), op)? {
            Some((value, _)) => Ok(Some(value)),
            None => Ok(None),
        }
    }

    /// Visits the attributes of the object from `cursor` onwards until `op` returns a
    /// value.
    ///
    /// Behaves like [`find_attr`](Self::find_attr). The value is returned together
    /// with the cursor of the next attribute, so the iteration can be resumed by
    /// passing that cursor back. Returns `None` once every attribute was visited,
    /// including when `cursor` is already at or past the last attribute. See
    /// [`Group::iter_visit_from`](crate::Group::iter_visit_from) for a paging loop.
    ///
    /// # Errors
    ///
    /// As for [`iter_attrs`](Self::iter_attrs).
    pub fn iter_attrs_from<B, F>(
        &self, cursor: IterationCursor, op: F,
    ) -> Result<Option<(B, IterationCursor)>>
    where
        F: FnMut(&str, AttrInfo) -> Result<Option<B>>,
    {
        visit(
            cursor,
            || Ok(self.loc_info()?.num_attrs as u64),
            op,
            |index_type, iteration_order, position, callback, op_data| unsafe {
                H5Aiterate2(
                    self.id(),
                    index_type,
                    iteration_order,
                    position,
                    Some(callback),
                    op_data,
                )
            },
        )
    }

    pub fn delete_attr(&self, name: &str) -> Result<()> {
        let name = to_cstring(name)?;
        h5call!(H5Adelete(self.id(), name.as_ptr()))?;
        Ok(())
    }

    /// Returns the object's metadata.
    pub fn loc_info(&self) -> Result<LocationInfo> {
        H5O_get_info(self.id(), true)
    }

    /// Returns the object's type.
    pub fn loc_type(&self) -> Result<LocationType> {
        Ok(H5O_get_info(self.id(), false)?.loc_type)
    }

    /// Returns the metadata of another object with name relative to `self`.
    ///
    /// # Errors
    ///
    /// Returns an error if the name is invalid.
    pub fn loc_info_by_name(&self, name: &str) -> Result<LocationInfo> {
        let name = to_cstring(name)?;
        H5O_get_info_by_name(self.id(), name.as_ptr(), true)
    }

    /// Returns the type of another object with name relative to `self`.
    ///
    /// # Errors
    ///
    /// Returns an error if the name is invalid.
    pub fn loc_type_by_name(&self, name: &str) -> Result<LocationType> {
        let name = to_cstring(name)?;
        Ok(H5O_get_info_by_name(self.id(), name.as_ptr(), false)?.loc_type)
    }

    /// Opens an object using its location token.
    pub fn open_by_token(&self, token: LocationToken) -> Result<Self> {
        H5O_open_by_token(self.id(), token)
    }

    /// Generate a [object reference](ObjectReference) to the object for a reference storage.
    ///
    /// This can be a group, dataset or datatype. Other objects are not supported.
    pub fn reference<R: ObjectReference>(&self, name: &str) -> Result<R> {
        R::create(self, name)
    }

    /// Get a reference back to the referenced object from a standard reference.
    ///
    /// This can be called against any object in the same file as the referenced object.
    pub fn dereference<R: ObjectReference>(&self, reference: &R) -> Result<ReferencedObject> {
        reference.dereference(self)
    }

    /// Copy this object to a destination location with default properties
    pub fn copy_to(&self, dst_loc: &Location, dst_name: &str) -> Result<()> {
        self.copy_to_with_props(dst_loc, dst_name, None, None)
    }

    /// Copy this object to a destination location with custom property lists
    ///
    /// # Arguments
    /// * `dst_loc` - Destination location (file or group)
    /// * `dst_name` - Name for the copied object at the destination
    /// * `ocpypl` - Optional object copy property list (controls copy behavior)
    /// * `lcpl` - Optional link creation property list (controls link properties)
    pub fn copy_to_with_props(
        &self, dst_loc: &Location, dst_name: &str, ocpypl: Option<&PropertyList>,
        lcpl: Option<&PropertyList>,
    ) -> Result<()> {
        // Validate property list classes if provided
        if let Some(pl) = ocpypl {
            if !pl.is_class(PropertyListClass::ObjectCopy) {
                fail!("Property list must be of class ObjectCopy");
            }
        }
        if let Some(pl) = lcpl {
            if !pl.is_class(PropertyListClass::LinkCreate) {
                fail!("Property list must be of class LinkCreate");
            }
        }

        let dst_name = to_cstring(dst_name)?;
        let ocpypl_id = ocpypl.map_or(H5P_DEFAULT, |p| p.id());
        let lcpl_id = lcpl.map_or(H5P_DEFAULT, |p| p.id());

        h5call!(H5Ocopy(
            self.id(),                        // src_loc_id
            b".\0".as_ptr() as *const c_char, // src_name (current object)
            dst_loc.id(),                     // dst_loc_id
            dst_name.as_ptr(),                // dst_name
            ocpypl_id,                        // ocpypl_id
            lcpl_id                           // lcpl_id
        ))?;
        Ok(())
    }
}

/// A token containing the address or identifier of a [`Location`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocationToken(
    #[cfg(not(feature = "1.12.0"))] haddr_t,
    #[cfg(feature = "1.12.0")] H5O_token_t,
);

/// The type of an object in a [`Location`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocationType {
    Group,
    Dataset,
    NamedDatatype,
    #[cfg(feature = "1.12.0")]
    #[cfg_attr(docsrs, doc(cfg(feature = "1.12.0")))]
    TypeMap,
}

impl From<H5O_type_t> for LocationType {
    fn from(loc_type: H5O_type_t) -> Self {
        // we're assuming here that if a C API call returns H5O_TYPE_UNKNOWN (-1), then
        // an error has occurred anyway and has been pushed on the error stack so we'll
        // catch it, and the value of -1 will never reach this conversion function
        match loc_type {
            H5O_type_t::H5O_TYPE_DATASET => Self::Dataset,
            H5O_type_t::H5O_TYPE_NAMED_DATATYPE => Self::NamedDatatype,
            #[cfg(feature = "1.12.0")]
            H5O_type_t::H5O_TYPE_MAP => Self::TypeMap,
            _ => Self::Group, // see the comment above
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Metadata information describing a [`Location`]
///
/// # Notes
///
/// In order for all timestamps to be filled out, a few conditions must hold:
///
/// - Minimum HDF5 library version is 1.10.3.
/// - Library version lower bound in the file access plist must be set to a least 1.10. This
///   can be done via `FileAccessBuilder::libver_v110` or `FileAccessBuilder::libver_latest`.
/// - For datasets, additionally, time tracking must be enabled (which is disabled
///   by default to improve access performance). This can be done via
///   `DatasetBuilder::track_times`. If tracking is enabled, ctime timestamp will likely be
///   filled out even if library version lower bound is not set), but the other three will
///   be zero.
pub struct LocationInfo {
    /// Number of file where the object is located
    pub fileno: u64,
    /// Object address in file, or a token identifier
    pub token: LocationToken,
    /// Basic location type of the object
    pub loc_type: LocationType,
    /// Number of hard links to the object
    pub num_links: usize,
    /// Access time
    pub atime: i64,
    /// Modification time
    pub mtime: i64,
    /// Change time
    pub ctime: i64,
    /// Birth time
    pub btime: i64,
    /// Number of attributes attached to the object
    pub num_attrs: usize,
}

#[cfg(not(feature = "1.12.0"))]
impl From<H5O_info1_t> for LocationInfo {
    fn from(info: H5O_info1_t) -> Self {
        Self {
            fileno: info.fileno as _,
            token: LocationToken(info.addr),
            loc_type: info.type_.into(),
            num_links: info.rc as _,
            atime: info.atime as _,
            mtime: info.mtime as _,
            ctime: info.ctime as _,
            btime: info.btime as _,
            num_attrs: info.num_attrs as _,
        }
    }
}

#[cfg(feature = "1.12.0")]
impl From<H5O_info2_t> for LocationInfo {
    fn from(info: H5O_info2_t) -> Self {
        Self {
            fileno: info.fileno as _,
            token: LocationToken(info.token),
            loc_type: info.type_.into(),
            num_links: info.rc as _,
            atime: info.atime as _,
            mtime: info.mtime as _,
            ctime: info.ctime as _,
            btime: info.btime as _,
            num_attrs: info.num_attrs as _,
        }
    }
}

#[cfg(feature = "1.10.3")]
fn info_fields(full: bool) -> c_uint {
    if full { H5O_INFO_BASIC | H5O_INFO_NUM_ATTRS | H5O_INFO_TIME } else { H5O_INFO_BASIC }
}

#[allow(non_snake_case, unused_variables)]
pub(crate) fn H5O_get_info(loc_id: hid_t, full: bool) -> Result<LocationInfo> {
    let mut info_buf = MaybeUninit::uninit();
    let info_ptr = info_buf.as_mut_ptr();
    #[cfg(feature = "1.12.0")]
    h5call!(H5Oget_info3(loc_id, info_ptr, info_fields(full)))?;
    #[cfg(all(feature = "1.10.3", not(feature = "1.12.0")))]
    h5call!(H5Oget_info2(loc_id, info_ptr, info_fields(full)))?;
    #[cfg(not(feature = "1.10.3"))]
    h5call!(H5Oget_info1(loc_id, info_ptr))?;
    let info = unsafe { info_buf.assume_init() };
    Ok(info.into())
}

#[allow(non_snake_case, unused_variables)]
fn H5O_get_info_by_name(loc_id: hid_t, name: *const c_char, full: bool) -> Result<LocationInfo> {
    let mut info_buf = MaybeUninit::uninit();
    let info_ptr = info_buf.as_mut_ptr();
    #[cfg(feature = "1.12.0")]
    h5call!(H5Oget_info_by_name3(loc_id, name, info_ptr, info_fields(full), H5P_DEFAULT))?;
    #[cfg(all(feature = "1.10.3", not(feature = "1.12.0")))]
    h5call!(H5Oget_info_by_name2(loc_id, name, info_ptr, info_fields(full), H5P_DEFAULT))?;
    #[cfg(not(feature = "1.10.3"))]
    h5call!(H5Oget_info_by_name1(loc_id, name, info_ptr, H5P_DEFAULT))?;
    let info = unsafe { info_buf.assume_init() };
    Ok(info.into())
}

#[allow(non_snake_case)]
fn H5O_open_by_token(loc_id: hid_t, token: LocationToken) -> Result<Location> {
    #[cfg(not(feature = "1.12.0"))]
    {
        Location::from_id(h5call!(H5Oopen_by_addr(loc_id, token.0))?)
    }
    #[cfg(feature = "1.12.0")]
    {
        Location::from_id(h5call!(H5Oopen_by_token(loc_id, token.0))?)
    }
}

#[cfg(test)]
pub mod tests {
    use crate::hl::plist::common::AttrCreationOrder;
    #[cfg(feature = "1.10.2")]
    use crate::hl::plist::file_access::LibraryVersion;
    use crate::hl::plist::link_create::CharEncoding;
    use crate::{hl::plist::object_copy::ObjectCopy, internal_prelude::*, plist::LinkCreate};

    #[test]
    pub fn test_filename() {
        with_tmp_path(|path| {
            assert_eq!(File::create(&path).unwrap().filename(), path.to_str().unwrap());
        })
    }

    #[test]
    pub fn test_name() {
        with_tmp_file(|file| {
            assert_eq!(file.name(), "/");
        })
    }

    #[test]
    pub fn test_file() {
        with_tmp_file(|file| {
            assert_eq!(file.file().unwrap().id(), file.id());
        })
    }

    #[test]
    pub fn test_comment() {
        #[allow(deprecated)]
        with_tmp_file(|file| {
            assert!(file.comment().is_none());
            assert!(file.set_comment("foo").is_ok());
            assert_eq!(file.comment().unwrap(), "foo");
            assert!(file.clear_comment().is_ok());
            assert!(file.comment().is_none());
        })
    }

    #[test]
    pub fn test_location_info() {
        let new_file = |path| {
            cfg_if::cfg_if! {
                if #[cfg(feature = "1.10.2")] {
                    File::with_options().with_fapl(|p| p.libver_v110()).create(path)
                } else {
                    File::create(path)
                }
            }
        };
        with_tmp_path(|path| {
            let file = new_file(path).unwrap();
            let token = {
                let group = file.create_group("group").unwrap();
                assert_eq!(file.loc_type_by_name("group").unwrap(), LocationType::Group);
                let info = group.loc_info().unwrap();
                assert_eq!(info.num_links, 1);
                assert_eq!(info.loc_type, LocationType::Group);
                cfg_if::cfg_if! {
                    if #[cfg(feature = "1.10.2")] {
                        assert!(info.btime > 0);
                    } else {
                        assert_eq!(info.btime, 0);
                    }
                }
                assert_eq!(info.btime == 0, info.mtime == 0);
                assert_eq!(info.btime == 0, info.ctime == 0);
                assert_eq!(info.btime == 0, info.atime == 0);
                assert_eq!(info.num_attrs, 0);
                info.token
            };
            let group = file.open_by_token(token).unwrap().as_group().unwrap();
            assert_eq!(group.name(), "/group");
            let token = {
                let var = group
                    .new_dataset_builder()
                    .obj_track_times(true)
                    .empty::<i8>()
                    .create("var")
                    .unwrap();
                var.new_attr::<i16>().create("attr1").unwrap();
                var.new_attr::<i32>().create("attr2").unwrap();
                group.link_hard("var", "hard1").unwrap();
                group.link_hard("var", "hard2").unwrap();
                group.link_hard("var", "hard3").unwrap();
                group.link_hard("var", "hard4").unwrap();
                group.link_hard("var", "hard5").unwrap();
                group.link_soft("var", "soft1").unwrap();
                group.link_soft("var", "soft2").unwrap();
                group.link_soft("var", "soft3").unwrap();
                assert_eq!(file.loc_type_by_name("/group/var").unwrap(), LocationType::Dataset);
                let info = var.loc_info().unwrap();
                assert_eq!(info.num_links, 6); // 1 + 5
                assert_eq!(info.loc_type, LocationType::Dataset);
                assert!(info.ctime > 0);
                cfg_if::cfg_if! {
                    if #[cfg(feature = "1.10.2")] {
                        assert!(info.btime > 0);
                    } else {
                        assert_eq!(info.btime, 0);
                    }
                }
                assert_eq!(info.btime == 0, info.mtime == 0);
                assert_eq!(info.btime == 0, info.atime == 0);
                assert_eq!(info.num_attrs, 2);
                info.token
            };
            let var = file.open_by_token(token).unwrap();
            // will open either the first or the last hard-linked object
            assert!(var.name().starts_with("/group/hard"));

            let info = file.loc_info_by_name("group").unwrap();
            let group = file.open_by_token(info.token).unwrap();
            assert_eq!(group.name(), "/group");
            let info = file.loc_info_by_name("/group/var").unwrap();
            let var = file.open_by_token(info.token).unwrap();
            assert!(var.name().starts_with("/group/hard"));

            assert!(file.loc_info_by_name("gibberish").is_err());
        })
    }

    #[test]
    pub fn test_copy_dataset_between_files() {
        with_tmp_path(|src_path| {
            with_tmp_path(|dst_path| {
                // Create source file with a dataset
                let src_file = File::create(&src_path).unwrap();
                let src_group = src_file.create_group("src_group").unwrap();

                let src_dataset =
                    src_group.new_dataset::<i32>().shape([5]).create("src_group").unwrap();
                src_dataset.write(&[1, 2, 3, 4, 5]).unwrap();

                let src_attr = src_dataset.new_attr::<f64>().create("src_attr").unwrap();
                src_attr.write_scalar(&42.0).unwrap();

                let dst_file = File::create(&dst_path).unwrap();
                let dst_group = dst_file.create_group("dst_group").unwrap();

                // Copy the dataset from source to destination and verify contents
                src_dataset.copy_to(&dst_group, "copied_dataset").unwrap();

                let copied_dataset = dst_group.dataset("copied_dataset").unwrap();
                let read_data: Vec<i32> = copied_dataset.read_1d().unwrap().to_vec();
                assert_eq!(read_data, &[1, 2, 3, 4, 5]);

                let copied_attr = copied_dataset.attr("src_attr").unwrap();
                let attr_value: f64 = copied_attr.read_scalar().unwrap();
                assert_eq!(attr_value, 42.0);
            })
        })
    }

    #[test]
    pub fn test_copy_group_with_nested_content() {
        with_tmp_path(|src_path| {
            with_tmp_path(|dst_path| {
                // Create source file with nested structure
                let src_file = File::create(&src_path).unwrap();
                let group1 = src_file.create_group("group1").unwrap();
                let subgroup = group1.create_group("subgroup").unwrap();

                let ds1 = group1.new_dataset::<i32>().shape([3]).create("dataset1").unwrap();
                ds1.write(&[10, 20, 30]).unwrap();
                let ds2 = subgroup.new_dataset::<f64>().shape([2]).create("dataset2").unwrap();
                ds2.write(&[1.5, 2.5]).unwrap();

                // Create destination file and copy entire group structure
                let dst_file = File::create(&dst_path).unwrap();
                group1.copy_to(&dst_file, "copied_group1").unwrap();

                // Verify the copied structure
                let copied_group = dst_file.group("copied_group1").unwrap();

                let copied_ds1 = copied_group.dataset("dataset1").unwrap();
                let data1: Vec<i32> = copied_ds1.read_1d().unwrap().to_vec();
                assert_eq!(data1, vec![10, 20, 30]);

                let copied_subgroup = copied_group.group("subgroup").unwrap();
                let copied_ds2 = copied_subgroup.dataset("dataset2").unwrap();
                let data2: Vec<f64> = copied_ds2.read_1d().unwrap().to_vec();
                assert_eq!(data2, vec![1.5, 2.5]);
            })
        })
    }

    #[test]
    pub fn test_copy_without_attributes() {
        with_tmp_path(|src_path| {
            with_tmp_path(|dst_path| {
                let src_file = File::create(&src_path).unwrap();

                let dataset = src_file.new_dataset::<i32>().shape([3]).create("data").unwrap();
                dataset.write(&[10, 20, 30]).unwrap();

                let src_attr = dataset.new_attr::<i32>().create("foo_attr").unwrap();
                src_attr.write_scalar(&100).unwrap();

                let dst_file = File::create(&dst_path).unwrap();

                let ocpypl = ObjectCopy::build().copy_without_attr(true).finish().unwrap();

                // Copy without attributes
                dataset
                    .copy_to_with_props(&dst_file, "copied_no_attrs", Some(&ocpypl), None)
                    .unwrap();

                src_file.close().unwrap();

                let copied = dst_file.dataset("copied_no_attrs").unwrap();
                let copied_data: Vec<i32> = copied.read_1d().unwrap().to_vec();
                assert_eq!(copied_data, vec![10, 20, 30]);

                // Verify attributes were NOT copied
                let attr_names = copied.attr_names().unwrap();
                assert!(attr_names.is_empty(), "Expected no attributes, but found: {attr_names:?}",);
            })
        })
    }

    #[test]
    pub fn test_copy_with_link_create_intermediate_groups() {
        with_tmp_path(|src_path| {
            with_tmp_path(|dst_path| {
                let src_file = File::create(&src_path).unwrap();

                let dataset = src_file.new_dataset::<i32>().shape([3]).create("data").unwrap();
                dataset.write(&[100, 200, 300]).unwrap();

                let dst_file = File::create(&dst_path).unwrap();

                // Fails without setting LinkCreate plist
                assert!(dataset.copy_to(&dst_file, "level1/level2/level3/copied_data").is_err());

                // Succeeds with LinkCreate
                // Create link create property list that creates intermediate groups
                dataset
                    .copy_to_with_props(
                        &dst_file,
                        "level1/level2/level3/copied_data",
                        None,
                        Some(
                            &LinkCreate::build().create_intermediate_group(true).finish().unwrap(),
                        ),
                    )
                    .unwrap();

                // Verify the intermediate groups were created
                assert!(dst_file.group("level1").is_ok());
                assert!(dst_file.group("level1/level2").is_ok());
                assert!(dst_file.group("level1/level2/level3").is_ok());

                // Verify the data
                let copied = dst_file.dataset("level1/level2/level3/copied_data").unwrap();
                let data: Vec<i32> = copied.read_1d().unwrap().to_vec();
                assert_eq!(data, vec![100, 200, 300]);
            })
        })
    }

    #[test]
    pub fn test_iter_attrs_order() {
        with_tmp_file(|file| {
            let obj = file.create_group("o").unwrap();
            for name in ["foo", "123", "bar"] {
                obj.new_attr::<u32>().create(name).unwrap();
            }
            let names = |order| obj.attr_names_by(IndexType::Name, order).unwrap();
            assert_eq!(names(IterationOrder::Increasing), ["123", "bar", "foo"]);
            assert_eq!(names(IterationOrder::Decreasing), ["foo", "bar", "123"]);
            assert_eq!(obj.attr_names().unwrap(), ["123", "bar", "foo"]);

            let empty = file.create_group("empty").unwrap();
            assert!(
                empty.attr_names_by(IndexType::Name, IterationOrder::Native).unwrap().is_empty()
            );
        })
    }

    #[test]
    pub fn test_iter_attrs_creation_order() {
        with_tmp_file(|file| {
            let obj = file
                .create_group_builder()
                .with_gcpl(|gcpl| gcpl.attr_creation_order(AttrCreationOrder::Tracked))
                .create("o")
                .unwrap();
            obj.new_attr::<u32>().create("foo").unwrap();
            obj.new_attr::<u64>().char_encoding(CharEncoding::Ascii).create("123").unwrap();
            obj.new_attr::<u32>().create("bar").unwrap();

            let attr = |name: &str, order, char_encoding, data_size| {
                (
                    name.to_owned(),
                    AttrInfo { creation_order: Some(order), char_encoding, data_size },
                )
            };
            let foo = attr("foo", 0, CharEncoding::Utf8, 4);
            let num = attr("123", 1, CharEncoding::Ascii, 8);
            let bar = attr("bar", 2, CharEncoding::Utf8, 4);
            let attrs = |order| obj.attrs(IndexType::CreationOrder, order).unwrap();
            assert_eq!(attrs(IterationOrder::Increasing), [foo.clone(), num.clone(), bar.clone()]);
            assert_eq!(attrs(IterationOrder::Decreasing), [bar, num, foo]);
        })
    }

    #[test]
    pub fn test_find_attr() {
        with_tmp_file(|file| {
            for name in ["a", "b", "c"] {
                file.new_attr::<u32>().create(name).unwrap();
            }

            let find = |wanted: &str| {
                let mut visited = vec![];
                let found = file
                    .find_attr(IndexType::Name, IterationOrder::Increasing, |name, info| {
                        visited.push(name.to_owned());
                        if name == wanted { Ok(Some(info.data_size)) } else { Ok(None) }
                    })
                    .unwrap();
                (found, visited)
            };

            let (found, visited) = find("b");
            assert_eq!(found, Some(4));
            assert_eq!(visited, ["a", "b"]);

            let (found, visited) = find("z");
            assert_eq!(found, None);
            assert_eq!(visited, ["a", "b", "c"]);
        })
    }

    #[test]
    pub fn test_iter_attrs_from() {
        with_tmp_file(|file| {
            for name in ["a", "b", "c"] {
                file.new_attr::<u32>().create(name).unwrap();
            }
            let start = IterationCursor::start(IndexType::Name, IterationOrder::Increasing);

            let stop_at = |cursor, wanted: &str| {
                let mut visited = vec![];
                let stopped = file
                    .iter_attrs_from(cursor, |name, _| {
                        visited.push(name.to_owned());
                        if name == wanted { Ok(Some(())) } else { Ok(None) }
                    })
                    .unwrap();
                (stopped, visited)
            };

            let (stopped, visited) = stop_at(start, "b");
            assert_eq!(visited, ["a", "b"]);
            assert_eq!(stopped, Some(((), start.skip(2))));

            let (stopped, visited) = stop_at(start.skip(2), "z");
            assert_eq!(visited, ["c"]);
            assert_eq!(stopped, None);

            let (stopped, visited) = stop_at(start.skip(3), "a");
            assert!(visited.is_empty());
            assert_eq!(stopped, None);
        })
    }

    #[test]
    pub fn test_attr_by_index() {
        with_tmp_file(|file| {
            let obj = file
                .create_group_builder()
                .with_gcpl(|gcpl| gcpl.attr_creation_order(AttrCreationOrder::Tracked))
                .create("o")
                .unwrap();
            for name in ["c", "a", "b"] {
                obj.new_attr::<u32>().create(name).unwrap();
            }

            let name = |index_type, iteration_order, index| {
                obj.attr_by_index(index_type, iteration_order, index).unwrap().name()
            };
            assert_eq!(name(IndexType::CreationOrder, IterationOrder::Increasing, 1), "a");
            assert_eq!(name(IndexType::CreationOrder, IterationOrder::Decreasing, 0), "b");
            assert_eq!(name(IndexType::Name, IterationOrder::Increasing, 2), "c");

            let err =
                obj.attr_by_index(IndexType::Name, IterationOrder::Increasing, 3).unwrap_err();
            assert!(err.contains_major(MajorErrorCode::Args), "{err:?}");
            assert!(err.contains_minor(MinorErrorCode::BadValue), "{err:?}");
        })
    }

    #[test]
    pub fn test_attr_info() {
        with_tmp_file(|file| {
            file.new_attr::<u32>().create("a").unwrap();
            file.new_attr::<u64>().char_encoding(CharEncoding::Ascii).create("b").unwrap();

            let expected = AttrInfo {
                creation_order: Some(1),
                char_encoding: CharEncoding::Ascii,
                data_size: 8,
            };
            assert_eq!(file.attr_info("b").unwrap(), expected);
            let reported = file
                .find_attr(IndexType::Name, IterationOrder::Increasing, |name, info| {
                    if name == "b" { Ok(Some(info)) } else { Ok(None) }
                })
                .unwrap();
            assert_eq!(reported, Some(expected));

            let err = file.attr_info("missing").unwrap_err();
            assert!(err.contains_major(MajorErrorCode::Attr), "{err:?}");
            assert!(err.contains_minor(MinorErrorCode::NotFound), "{err:?}");
        })
    }

    fn attrs_by_creation_order(
        file: &File, name: &str, creation_order: AttrCreationOrder, dense: bool,
    ) -> Vec<(String, Option<u32>)> {
        let obj = file
            .create_group_builder()
            .with_gcpl(|gcpl| {
                gcpl.attr_creation_order(creation_order);
                if dense {
                    gcpl.attr_phase_change(0, 0);
                }
                gcpl
            })
            .create(name)
            .unwrap();
        for name in ["c", "a", "b"] {
            obj.new_attr::<u32>().create(name).unwrap();
        }
        let attrs = obj.attrs(IndexType::CreationOrder, IterationOrder::Increasing).unwrap();
        attrs.into_iter().map(|(name, info)| (name, info.creation_order)).collect()
    }

    fn assert_storage_sequence(attrs: &[(String, Option<u32>)]) {
        let expected =
            [("c".to_owned(), Some(0)), ("a".to_owned(), Some(1)), ("b".to_owned(), Some(2))];
        assert_eq!(attrs, expected);
    }

    fn assert_name_index_fallback(attrs: &[(String, Option<u32>)]) {
        let mut names: Vec<&str> = attrs.iter().map(|(name, _)| name.as_str()).collect();
        names.sort_unstable();
        assert_eq!(names, ["a", "b", "c"]);
        assert!(attrs.iter().all(|(_, order)| order.is_none()), "{attrs:?}");
    }

    // Only a tracked object has an attribute creation order index. Compact storage in
    // the object header still yields the storage sequence with positions, dense storage
    // in a heap falls back to the name index. Dense storage needs a version-2 object
    // header, which the default file format only produces from libhdf5 2.0 on.
    #[test]
    pub fn test_iter_attrs_untracked_default_format() {
        with_tmp_file(|file| {
            let compact =
                attrs_by_creation_order(&file, "compact", AttrCreationOrder::Untracked, false);
            assert_storage_sequence(&compact);

            let dense = attrs_by_creation_order(&file, "dense", AttrCreationOrder::Untracked, true);
            if cfg!(feature = "2.0.0") {
                assert_name_index_fallback(&dense);
            } else {
                assert_storage_sequence(&dense);
            }
        })
    }

    #[cfg(feature = "1.10.2")]
    #[test]
    pub fn test_iter_attrs_v2_object_header() {
        for low in [LibraryVersion::V18, LibraryVersion::latest()] {
            with_tmp_path(|path| {
                let file = File::with_options()
                    .with_fapl(|fapl| fapl.libver_bounds(low, LibraryVersion::latest()))
                    .create(&path)
                    .unwrap();
                let attrs = |name, creation_order, dense| {
                    attrs_by_creation_order(&file, name, creation_order, dense)
                };
                assert_storage_sequence(&attrs(
                    "tracked_compact",
                    AttrCreationOrder::Tracked,
                    false,
                ));
                assert_storage_sequence(&attrs("tracked_dense", AttrCreationOrder::Tracked, true));
                assert_storage_sequence(&attrs(
                    "untracked_compact",
                    AttrCreationOrder::Untracked,
                    false,
                ));
                assert_name_index_fallback(&attrs(
                    "untracked_dense",
                    AttrCreationOrder::Untracked,
                    true,
                ));
            })
        }
    }
}
