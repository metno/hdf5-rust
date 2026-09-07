use std::fmt::{self, Debug};
use std::ops::Deref;

use hdf5_sys::{
    h5d::H5Dopen2,
    h5g::{
        H5G_info_t, H5G_storage_type_t, H5Gcreate_anon, H5Gcreate2, H5Gget_create_plist,
        H5Gget_info, H5Gopen2,
    },
    h5l::{
        H5L_SAME_LOC, H5L_info_t, H5L_type_t, H5Lcreate_external, H5Lcreate_hard, H5Lcreate_soft,
        H5Ldelete, H5Lexists, H5Literate, H5Lmove,
    },
    h5p::{H5Pcreate, H5Pset_create_intermediate_group},
    h5t::{H5Tcommit2, H5Topen2},
};

use crate::globals::H5P_LINK_CREATE;
use crate::hl::dataset::Maybe;
use crate::hl::iteration::visit;
use crate::hl::plist::group_create::{GroupCreate, GroupCreateBuilder};
use crate::hl::plist::link_create::{CharEncoding, LinkCreate, LinkCreateBuilder};
use crate::internal_prelude::*;
use crate::{Location, LocationType};

/// Represents the HDF5 group object.
#[repr(transparent)]
#[derive(Clone)]
pub struct Group(Handle);

impl ObjectClass for Group {
    const NAME: &'static str = "group";
    const VALID_TYPES: &'static [H5I_type_t] = &[H5I_GROUP, H5I_FILE];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }

    fn short_repr(&self) -> Option<String> {
        let members = match self.len() {
            0 => "empty".to_owned(),
            1 => "1 member".to_owned(),
            x => format!("{x} members"),
        };
        Some(format!("\"{}\" ({})", self.name(), members))
    }
}

impl Debug for Group {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.debug_fmt(f)
    }
}

impl Deref for Group {
    type Target = Location;

    fn deref(&self) -> &Location {
        unsafe { self.transmute() }
    }
}

fn group_info(id: hid_t) -> Result<H5G_info_t> {
    let info: *mut H5G_info_t = &mut H5G_info_t::default();
    h5call!(H5Gget_info(id, info)).and(Ok(unsafe { *info }))
}

fn make_lcpl() -> Result<PropertyList> {
    h5lock!({
        let lcpl = PropertyList::from_id(h5try!(H5Pcreate(*H5P_LINK_CREATE)))?;
        h5call!(H5Pset_create_intermediate_group(lcpl.id(), 1)).and(Ok(lcpl))
    })
}

impl Group {
    /// Returns the number of objects in the container (or 0 if the container is invalid).
    pub fn len(&self) -> u64 {
        group_info(self.id()).map(|info| info.nlinks).unwrap_or(0)
    }

    /// Returns true if the container has no linked objects (or if the container is invalid).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns information about the group.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_metno::plist::group_create::LinkCreationOrder;
    /// use hdf5_metno::{File, GroupStorageType};
    ///
    /// let file = File::with_options().with_fapl(|p| p.core_filebacked(false)).create("group_info.h5")?;
    /// let group = file
    ///     .create_group_builder()
    ///     .with_gcpl(|gcpl| gcpl.link_creation_order(LinkCreationOrder::Tracked))
    ///     .create("g")?;
    /// group.create_group("a")?;
    /// group.create_group("b")?;
    /// group.unlink("a")?;
    ///
    /// let info = group.info()?;
    /// assert_eq!(info.storage_type, GroupStorageType::Compact);
    /// assert_eq!(info.nlinks, 1);
    /// assert_eq!(info.max_corder, 2);
    /// # Ok::<(), hdf5_metno::Error>(())
    /// ```
    pub fn info(&self) -> Result<GroupInfo> {
        group_info(self.id())?.try_into()
    }

    /// Create a new group in a file or group.
    pub fn create_group(&self, name: &str) -> Result<Self> {
        // TODO: &mut self?
        self.create_group_builder().create(name)
    }

    /// Instantiates a new group builder for configuring group creation properties.
    ///
    /// Intermediate groups are created automatically (as with [`create_group`](Self::create_group))
    /// unless disabled via [`GroupBuilder::create_intermediate_group`].
    pub fn create_group_builder(&self) -> GroupBuilder {
        GroupBuilder::new(self)
    }

    /// Opens an existing group in a file or group.
    pub fn group(&self, name: &str) -> Result<Self> {
        let name = to_cstring(name)?;
        Self::from_id(h5try!(H5Gopen2(self.id(), name.as_ptr(), H5P_DEFAULT)))
    }

    /// Creates a soft link.
    ///
    /// A soft link does not require the linked object to exist.
    /// Note: `target` and `link_name` are relative to the current object.
    pub fn link_soft(&self, target: &str, link_name: &str) -> Result<()> {
        // TODO: &mut self?
        h5lock!({
            let lcpl = make_lcpl()?;
            let target = to_cstring(target)?;
            let link_name = to_cstring(link_name)?;
            h5call!(H5Lcreate_soft(
                target.as_ptr(),
                self.id(),
                link_name.as_ptr(),
                lcpl.id(),
                H5P_DEFAULT
            ))
            .and(Ok(()))
        })
    }

    /// Creates a hard link. Note: `target` and `link_name` are relative to the current object.
    pub fn link_hard(&self, target: &str, link_name: &str) -> Result<()> {
        // TODO: &mut self?
        let target = to_cstring(target)?;
        let link_name = to_cstring(link_name)?;
        h5call!(H5Lcreate_hard(
            self.id(),
            target.as_ptr(),
            H5L_SAME_LOC,
            link_name.as_ptr(),
            H5P_DEFAULT,
            H5P_DEFAULT
        ))
        .and(Ok(()))
    }

    /// Creates an external link.
    ///
    /// Note: `link_name` is relative to the current object,
    /// `target` is relative to the root of the source file,
    /// `target_file_name` is the path to the external file.
    ///
    /// For a detailed explanation on how `target_file_name` is resolved, see
    /// [https://portal.hdfgroup.org/display/HDF5/H5L_CREATE_EXTERNAL](https://portal.hdfgroup.org/display/HDF5/H5L_CREATE_EXTERNAL)
    pub fn link_external(
        &self, target_file_name: &str, target: &str, link_name: &str,
    ) -> Result<()> {
        // TODO: &mut self?
        let target = to_cstring(target)?;
        let target_file_name = to_cstring(target_file_name)?;
        let link_name = to_cstring(link_name)?;
        h5call!(H5Lcreate_external(
            target_file_name.as_ptr(),
            target.as_ptr(),
            self.id(),
            link_name.as_ptr(),
            H5P_DEFAULT,
            H5P_DEFAULT,
        ))
        .and(Ok(()))
    }

    /// Relinks an object. Note: `name` and `path` are relative to the current object.
    pub fn relink(&self, name: &str, path: &str) -> Result<()> {
        // TODO: &mut self?
        let name = to_cstring(name)?;
        let path = to_cstring(path)?;
        h5call!(H5Lmove(
            self.id(),
            name.as_ptr(),
            H5L_SAME_LOC,
            path.as_ptr(),
            H5P_DEFAULT,
            H5P_DEFAULT
        ))
        .and(Ok(()))
    }

    /// Removes a link to an object from this file or group.
    pub fn unlink(&self, name: &str) -> Result<()> {
        // TODO: &mut self?
        let name = to_cstring(name)?;
        h5call!(H5Ldelete(self.id(), name.as_ptr(), H5P_DEFAULT)).and(Ok(()))
    }

    /// Check if a link with a given name exists in this file or group.
    pub fn link_exists(&self, name: &str) -> bool {
        (|| -> Result<bool> {
            let name = to_cstring(name)?;
            Ok(h5call!(H5Lexists(self.id(), name.as_ptr(), H5P_DEFAULT))? > 0)
        })()
        .unwrap_or(false)
    }

    /// Instantiates a new typed dataset builder.
    pub fn new_dataset<T: H5Type>(&self) -> DatasetBuilderEmpty {
        self.new_dataset_builder().empty::<T>()
    }

    /// Instantiates a new dataset builder.
    pub fn new_dataset_builder(&self) -> DatasetBuilder {
        DatasetBuilder::new(self)
    }

    /// Opens an existing dataset in the file or group.
    pub fn dataset(&self, name: &str) -> Result<Dataset> {
        let name = to_cstring(name)?;
        Dataset::from_id(h5try!(H5Dopen2(self.id(), name.as_ptr(), H5P_DEFAULT)))
    }

    /// Returns a copy of the group creation property list.
    pub fn create_plist(&self) -> Result<GroupCreate> {
        h5lock!(GroupCreate::from_id(h5try!(H5Gget_create_plist(self.id()))))
    }

    /// A short alias for `create_plist()`.
    pub fn gcpl(&self) -> Result<GroupCreate> {
        self.create_plist()
    }
}

/// A builder for creating a new [`Group`].
///
/// Created via [`Group::create_group_builder`]. Allows configuring the group
/// creation property list (e.g. [`obj_track_times`](Self::obj_track_times)) and the
/// link creation property list before creating the group.
#[derive(Clone)]
pub struct GroupBuilder {
    parent: Result<Handle>,
    gcpl_base: Option<GroupCreate>,
    gcpl_builder: GroupCreateBuilder,
    lcpl_base: Option<LinkCreate>,
    lcpl_builder: LinkCreateBuilder,
}

impl GroupBuilder {
    /// Creates a new group builder with the given parent location.
    pub fn new(parent: &Group) -> Self {
        // enable creation of intermediate groups by default, matching `create_group`
        let mut lcpl_builder = LinkCreateBuilder::default();
        lcpl_builder.create_intermediate_group(true);
        Self {
            parent: parent.try_borrow(),
            gcpl_base: None,
            gcpl_builder: GroupCreateBuilder::default(),
            lcpl_base: None,
            lcpl_builder,
        }
    }

    /// Uses an existing group creation property list as the base.
    #[inline]
    #[must_use]
    pub fn set_create_plist(mut self, gcpl: &GroupCreate) -> Self {
        self.gcpl_base = Some(gcpl.clone());
        self
    }

    /// Alias for [`set_create_plist`](Self::set_create_plist).
    #[inline]
    #[must_use]
    pub fn set_gcpl(self, gcpl: &GroupCreate) -> Self {
        self.set_create_plist(gcpl)
    }

    /// Returns a mutable reference to the group creation property list builder.
    #[inline]
    pub fn create_plist(&mut self) -> &mut GroupCreateBuilder {
        &mut self.gcpl_builder
    }

    /// Alias for [`create_plist`](Self::create_plist).
    #[inline]
    pub fn gcpl(&mut self) -> &mut GroupCreateBuilder {
        self.create_plist()
    }

    /// Applies a closure to the group creation property list builder.
    #[inline]
    #[must_use]
    pub fn with_create_plist<F>(mut self, func: F) -> Self
    where
        F: Fn(&mut GroupCreateBuilder) -> &mut GroupCreateBuilder,
    {
        func(&mut self.gcpl_builder);
        self
    }

    /// Alias for [`with_create_plist`](Self::with_create_plist).
    #[inline]
    #[must_use]
    pub fn with_gcpl<F>(self, func: F) -> Self
    where
        F: Fn(&mut GroupCreateBuilder) -> &mut GroupCreateBuilder,
    {
        self.with_create_plist(func)
    }

    #[inline]
    #[must_use]
    #[doc = "\u{21b3} [`GroupCreateBuilder::obj_track_times`](crate::plist::GroupCreateBuilder::obj_track_times)"]
    pub fn obj_track_times(mut self, track_times: bool) -> Self {
        self.gcpl_builder.obj_track_times(track_times);
        self
    }

    /// Uses an existing link creation property list as the base.
    #[inline]
    #[must_use]
    pub fn set_link_create_plist(mut self, lcpl: &LinkCreate) -> Self {
        self.lcpl_base = Some(lcpl.clone());
        self
    }

    /// Alias for [`set_link_create_plist`](Self::set_link_create_plist).
    #[inline]
    #[must_use]
    pub fn set_lcpl(self, lcpl: &LinkCreate) -> Self {
        self.set_link_create_plist(lcpl)
    }

    /// Returns a mutable reference to the link creation property list builder.
    #[inline]
    pub fn link_create_plist(&mut self) -> &mut LinkCreateBuilder {
        &mut self.lcpl_builder
    }

    /// Alias for [`link_create_plist`](Self::link_create_plist).
    #[inline]
    pub fn lcpl(&mut self) -> &mut LinkCreateBuilder {
        self.link_create_plist()
    }

    /// Applies a closure to the link creation property list builder.
    #[inline]
    #[must_use]
    pub fn with_link_create_plist<F>(mut self, func: F) -> Self
    where
        F: Fn(&mut LinkCreateBuilder) -> &mut LinkCreateBuilder,
    {
        func(&mut self.lcpl_builder);
        self
    }

    /// Alias for [`with_link_create_plist`](Self::with_link_create_plist).
    #[inline]
    #[must_use]
    pub fn with_lcpl<F>(self, func: F) -> Self
    where
        F: Fn(&mut LinkCreateBuilder) -> &mut LinkCreateBuilder,
    {
        self.with_link_create_plist(func)
    }

    #[inline]
    #[must_use]
    #[doc = "\u{21b3} [`LinkCreateBuilder::create_intermediate_group`](crate::plist::LinkCreateBuilder::create_intermediate_group)"]
    pub fn create_intermediate_group(mut self, create: bool) -> Self {
        self.lcpl_builder.create_intermediate_group(create);
        self
    }

    #[inline]
    #[must_use]
    #[doc = "\u{21b3} [`LinkCreateBuilder::char_encoding`](crate::plist::LinkCreateBuilder::char_encoding)"]
    pub fn char_encoding(mut self, encoding: CharEncoding) -> Self {
        self.lcpl_builder.char_encoding(encoding);
        self
    }

    fn build_gcpl(&self) -> Result<GroupCreate> {
        let mut gcpl = match &self.gcpl_base {
            Some(gcpl) => gcpl.clone(),
            None => GroupCreate::try_new()?,
        };
        self.gcpl_builder.apply(&mut gcpl).map(|()| gcpl)
    }

    fn build_lcpl(&self) -> Result<LinkCreate> {
        let mut lcpl = match &self.lcpl_base {
            Some(lcpl) => lcpl.clone(),
            None => LinkCreate::try_new()?,
        };
        self.lcpl_builder.apply(&mut lcpl).map(|()| lcpl)
    }

    /// Creates the group.
    ///
    /// Passing a name creates a named (linked) group. Passing `None` creates an
    /// anonymous group that is not linked into the file until linked explicitly.
    pub fn create<'n, T: Into<Maybe<&'n str>>>(&self, name: T) -> Result<Group> {
        h5lock!({
            let parent = try_ref_clone!(self.parent);
            let gcpl = self.build_gcpl()?;
            let name: Option<&str> = name.into().into();
            if let Some(name) = name {
                let lcpl = self.build_lcpl()?;
                let name = to_cstring(name)?;
                Group::from_id(h5try!(H5Gcreate2(
                    parent.id(),
                    name.as_ptr(),
                    lcpl.id(),
                    gcpl.id(),
                    H5P_DEFAULT
                )))
            } else {
                Group::from_id(h5try!(H5Gcreate_anon(parent.id(), gcpl.id(), H5P_DEFAULT)))
            }
        })
    }
}

/// How the links of a group are stored.
///
/// Corresponds to `H5G_storage_type_t`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupStorageType {
    /// Links in a symbol table, the original group format.
    SymbolTable,
    /// Links as messages in the object header.
    Compact,
    /// Links in a fractal heap with B-tree indexes.
    Dense,
}

/// Information about a group.
///
/// Corresponds to `H5G_info_t`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupInfo {
    /// How the links are stored.
    pub storage_type: GroupStorageType,
    /// Number of links in the group.
    pub nlinks: u64,
    /// Creation order position the next link receives. Unlinking does not lower it, so
    /// it counts every link ever created in a group that tracks link creation order.
    /// Zero in a group that does not track it.
    pub max_corder: i64,
    /// Whether a file is mounted on the group.
    pub mounted: bool,
}

impl TryFrom<H5G_info_t> for GroupInfo {
    type Error = Error;

    fn try_from(info: H5G_info_t) -> Result<Self> {
        let storage_type = match info.storage_type {
            H5G_storage_type_t::H5G_STORAGE_TYPE_SYMBOL_TABLE => GroupStorageType::SymbolTable,
            H5G_storage_type_t::H5G_STORAGE_TYPE_COMPACT => GroupStorageType::Compact,
            H5G_storage_type_t::H5G_STORAGE_TYPE_DENSE => GroupStorageType::Dense,
            storage_type => fail!("Unknown group storage type: {:?}", storage_type),
        };
        Ok(Self {
            storage_type,
            nlinks: info.nlinks,
            max_corder: info.max_corder,
            mounted: info.mounted > 0,
        })
    }
}

/// The type of an object link.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkType {
    /// A hard link to an object within a single file.
    Hard,
    /// A symbolic link to an object within a single file.
    Soft,
    /// A symbolic link to an object in a different file.
    External,
}

impl From<H5L_type_t> for LinkType {
    fn from(link_type: H5L_type_t) -> Self {
        match link_type {
            H5L_type_t::H5L_TYPE_HARD => Self::Hard,
            H5L_type_t::H5L_TYPE_SOFT => Self::Soft,
            _ => Self::External,
        }
    }
}

/// Metadata describing an object link.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinkInfo {
    pub link_type: LinkType,
    pub creation_order: Option<i64>,
    /// Encoding of the link name. HDF5 values other than UTF-8 are reported as ASCII.
    pub char_encoding: CharEncoding,
}

impl From<&H5L_info_t> for LinkInfo {
    fn from(link: &H5L_info_t) -> Self {
        let link_type = link.type_.into();
        let creation_order = if link.corder_valid == 1 { Some(link.corder) } else { None };
        let char_encoding = CharEncoding::try_from(link.cset).unwrap_or(CharEncoding::Ascii);
        Self { link_type, creation_order, char_encoding }
    }
}

/// Iteration methods
impl Group {
    /// Visits every link of the group, non-recursively.
    ///
    /// The links are traversed along `index_type` in `iteration_order`, and `op` is
    /// called with the name and the [`LinkInfo`] of each link. Use
    /// [`find_link`](Self::find_link) to stop early.
    ///
    /// # Errors
    ///
    /// Returns the first error returned by `op`. Returns the HDF5 error if the
    /// iteration itself fails, for example when traversing by
    /// [`IndexType::CreationOrder`] on a group that does not track link creation order.
    ///
    /// # Panics
    ///
    /// A panic in `op` is caught while HDF5 frames are on the stack and resumed once
    /// the iteration has returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_metno::{File, IndexType, IterationOrder};
    ///
    /// let file = File::with_options().with_fapl(|p| p.core_filebacked(false)).create("iter_visit.h5")?;
    /// file.create_group("b")?;
    /// file.create_group("a")?;
    ///
    /// let mut names = vec![];
    /// file.iter_visit(IndexType::Name, IterationOrder::Decreasing, |name, _| {
    ///     names.push(name.to_owned());
    ///     Ok(())
    /// })?;
    /// assert_eq!(names, ["b", "a"]);
    /// # Ok::<(), hdf5_metno::Error>(())
    /// ```
    pub fn iter_visit<F>(
        &self, index_type: IndexType, iteration_order: IterationOrder, mut op: F,
    ) -> Result<()>
    where
        F: FnMut(&str, LinkInfo) -> Result<()>,
    {
        self.iter_visit_from(IterationCursor::start(index_type, iteration_order), |name, info| {
            op(name, info)?;
            Ok(None::<()>)
        })?;
        Ok(())
    }

    /// Visits every link of the group by name in native order.
    ///
    /// Equivalent to [`iter_visit`](Self::iter_visit) with [`IndexType::Name`] and
    /// [`IterationOrder::Native`].
    pub fn iter_visit_default<F>(&self, op: F) -> Result<()>
    where
        F: FnMut(&str, LinkInfo) -> Result<()>,
    {
        self.iter_visit(IndexType::default(), IterationOrder::default(), op)
    }

    /// Visits the links of the group until `op` returns a value.
    ///
    /// The links are traversed along `index_type` in `iteration_order`, and `op` is
    /// called with the name and the [`LinkInfo`] of each link until it returns `Some`.
    /// That value is returned, or `None` once every link was visited.
    ///
    /// # Errors
    ///
    /// As for [`iter_visit`](Self::iter_visit).
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_metno::{File, IndexType, IterationOrder, LinkType};
    ///
    /// let file = File::with_options().with_fapl(|p| p.core_filebacked(false)).create("find_link.h5")?;
    /// file.create_group("b")?;
    /// file.link_soft("b", "a")?;
    ///
    /// let first_hard = file.find_link(IndexType::Name, IterationOrder::Increasing, |name, info| {
    ///     Ok((info.link_type == LinkType::Hard).then(|| name.to_owned()))
    /// })?;
    /// assert_eq!(first_hard, Some("b".to_owned()));
    /// # Ok::<(), hdf5_metno::Error>(())
    /// ```
    pub fn find_link<B, F>(
        &self, index_type: IndexType, iteration_order: IterationOrder, op: F,
    ) -> Result<Option<B>>
    where
        F: FnMut(&str, LinkInfo) -> Result<Option<B>>,
    {
        match self.iter_visit_from(IterationCursor::start(index_type, iteration_order), op)? {
            Some((value, _)) => Ok(Some(value)),
            None => Ok(None),
        }
    }

    /// Visits the links of the group from `cursor` onwards until `op` returns a value.
    ///
    /// Behaves like [`find_link`](Self::find_link). The value is returned together
    /// with the cursor of the next link, so the iteration can be resumed by passing
    /// that cursor back. Returns `None` once every link was visited, including when
    /// `cursor` is already at or past the last link.
    ///
    /// # Errors
    ///
    /// As for [`iter_visit`](Self::iter_visit).
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_metno::{File, IndexType, IterationOrder, IterationCursor};
    ///
    /// let file = File::with_options().with_fapl(|p| p.core_filebacked(false)).create("iter_visit_from.h5")?;
    /// for name in ["a", "b", "c"] {
    ///     file.create_group(name)?;
    /// }
    ///
    /// let mut cursor = IterationCursor::start(IndexType::Name, IterationOrder::Increasing).skip(1);
    /// let mut names = vec![];
    /// while let Some((name, next)) =
    ///     file.iter_visit_from(cursor, |name, _| Ok(Some(name.to_owned())))?
    /// {
    ///     names.push(name);
    ///     cursor = next;
    /// }
    /// assert_eq!(names, ["b", "c"]);
    /// assert_eq!(cursor.position(), 3);
    /// # Ok::<(), hdf5_metno::Error>(())
    /// ```
    pub fn iter_visit_from<B, F>(
        &self, cursor: IterationCursor, op: F,
    ) -> Result<Option<(B, IterationCursor)>>
    where
        F: FnMut(&str, LinkInfo) -> Result<Option<B>>,
    {
        visit(
            cursor,
            || Ok(group_info(self.id())?.nlinks),
            op,
            |index_type, iteration_order, position, callback, op_data| unsafe {
                H5Literate(
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

    fn get_all_of_type(&self, loc_type: LocationType) -> Result<Vec<Location>> {
        let mut objects = vec![];
        self.iter_visit_default(|name, _| {
            let info = self.loc_info_by_name(name)?;
            if info.loc_type == loc_type {
                objects.push(self.open_by_token(info.token)?);
            }
            Ok(())
        })?;
        Ok(objects)
    }

    /// Returns all groups in the group, non-recursively
    pub fn groups(&self) -> Result<Vec<Self>> {
        self.get_all_of_type(LocationType::Group)
            .map(|vec| vec.into_iter().map(|obj| unsafe { obj.cast_unchecked() }).collect())
    }

    /// Returns all datasets in the group, non-recursively
    pub fn datasets(&self) -> Result<Vec<Dataset>> {
        self.get_all_of_type(LocationType::Dataset)
            .map(|vec| vec.into_iter().map(|obj| unsafe { obj.cast_unchecked() }).collect())
    }

    /// Returns all committed datatypes in the group, non-recursively.
    pub fn committed_datatypes(&self) -> Result<Vec<CommittedDatatype>> {
        self.get_all_of_type(LocationType::NamedDatatype)
            .map(|vec| vec.into_iter().map(|obj| unsafe { obj.cast_unchecked() }).collect())
    }

    /// Returns all committed datatypes in the group, non-recursively.
    #[deprecated(note = "pre-1.12 HDF5 term for a committed datatype, use committed_datatypes()")]
    pub fn named_datatypes(&self) -> Result<Vec<CommittedDatatype>> {
        self.committed_datatypes()
    }

    /// Commits `datatype` as a committed datatype at `name`, relative to this group.
    ///
    /// On success `datatype` itself becomes the committed datatype: [`Datatype::is_committed`]
    /// reports `true` on it afterwards, and committing it a second time fails.
    pub fn commit_datatype(&self, name: &str, datatype: &Datatype) -> Result<()> {
        let name = to_cstring(name)?;
        h5call!(H5Tcommit2(
            self.id(),
            name.as_ptr(),
            datatype.id(),
            H5P_DEFAULT,
            H5P_DEFAULT,
            H5P_DEFAULT
        ))?;
        Ok(())
    }

    /// Opens the committed datatype at `name`, relative to this group.
    pub fn committed_datatype(&self, name: &str) -> Result<CommittedDatatype> {
        let name = to_cstring(name)?;
        CommittedDatatype::from_id(h5try!(H5Topen2(self.id(), name.as_ptr(), H5P_DEFAULT)))
    }

    /// Returns the names of all links in the group by name in native order, non-recursively.
    pub fn member_names(&self) -> Result<Vec<String>> {
        self.member_names_by(IndexType::default(), IterationOrder::default())
    }

    /// Returns the names of all links in the group along `index_type` in `iteration_order`.
    pub fn member_names_by(
        &self, index_type: IndexType, iteration_order: IterationOrder,
    ) -> Result<Vec<String>> {
        let mut names = vec![];
        self.iter_visit(index_type, iteration_order, |name, _| {
            names.push(name.to_owned());
            Ok(())
        })?;
        Ok(names)
    }

    /// Returns the name and [`LinkInfo`] of all links in the group along `index_type` in
    /// `iteration_order`.
    pub fn links(
        &self, index_type: IndexType, iteration_order: IterationOrder,
    ) -> Result<Vec<(String, LinkInfo)>> {
        let mut links = vec![];
        self.iter_visit(index_type, iteration_order, |name, info| {
            links.push((name.to_owned(), info));
            Ok(())
        })?;
        Ok(links)
    }
}

#[cfg(test)]
pub mod tests {
    use crate::hl::plist::common::{AttrCreationOrder, AttrPhaseChange, LinkCreationOrder};
    use crate::hl::plist::file_access::FileCloseDegree;
    #[cfg(feature = "1.10.2")]
    use crate::hl::plist::file_access::LibraryVersion;
    use crate::hl::plist::link_create::CharEncoding;
    use crate::internal_prelude::*;
    use crate::{IndexType, IterationCursor, IterationOrder, LinkType};
    use hdf5_types::{IntSize, TypeDescriptor, VarLenUnicode};
    use std::panic::{self, AssertUnwindSafe};

    #[test]
    pub fn test_debug() {
        with_tmp_path(|path| {
            let file = File::with_options()
                .with_fapl(|fapl| fapl.fclose_degree(FileCloseDegree::Strong))
                .create(&path)
                .unwrap();
            file.create_group("a/b/c").unwrap();
            file.create_group("/a/d").unwrap();
            let a = file.group("a").unwrap();
            let ab = file.group("/a/b").unwrap();
            let abc = file.group("./a/b/c/").unwrap();
            assert_eq!(format!("{:?}", a), "<HDF5 group: \"/a\" (2 members)>");
            assert_eq!(format!("{:?}", ab), "<HDF5 group: \"/a/b\" (1 member)>");
            assert_eq!(format!("{:?}", abc), "<HDF5 group: \"/a/b/c\" (empty)>");
            h5lock!({
                file.close().unwrap();
                assert_eq!(format!("{:?}", a), "<HDF5 group: invalid id>");
                drop(a);
                drop(ab);
                drop(abc);
            })
        })
    }

    #[test]
    pub fn test_group() {
        with_tmp_file(|file| {
            assert_err_re!(
                file.group("a"),
                "unable to (?:synchronously )?open group: object.+doesn't exist"
            );
            file.create_group("a").unwrap();
            let a = file.group("a").unwrap();
            assert_eq!(a.name(), "/a");
            assert_eq!(a.file().unwrap().id(), file.id());
            a.create_group("b").unwrap();
            let b = file.group("/a/b").unwrap();
            assert_eq!(b.name(), "/a/b");
            assert_eq!(b.file().unwrap().id(), file.id());
            file.create_group("/foo/bar").unwrap();
            file.group("foo").unwrap().group("bar").unwrap();
            file.create_group("x/y/").unwrap();
            file.group("/x").unwrap().group("./y/").unwrap();
        })
    }

    #[test]
    pub fn test_committed_datatype() {
        with_tmp_path(|path| {
            let dtype = Datatype::from_type::<i32>().unwrap();
            {
                let file = File::create(&path).unwrap();
                let err = file.committed_datatype("mytype").unwrap_err();
                assert!(err.contains_major(MajorErrorCode::Datatype), "{err:?}");
                assert!(err.contains_minor(MinorErrorCode::NotFound), "{err:?}");

                file.commit_datatype("mytype", &dtype).unwrap();
                assert!(dtype.is_committed());

                // 1.12+ rejects a committed handle in H5Tcommit2 (CANTSET), older versions in
                // H5T__commit (BADVALUE)
                let already_committed = if cfg!(feature = "1.12.0") {
                    MinorErrorCode::CantSet
                } else {
                    MinorErrorCode::BadValue
                };
                let err = file.commit_datatype("again", &dtype).unwrap_err();
                assert!(err.contains_major(MajorErrorCode::Args), "{err:?}");
                assert!(err.contains_minor(already_committed), "{err:?}");
                let fresh = Datatype::from_type::<i32>().unwrap();
                let err = file.commit_datatype("mytype", &fresh).unwrap_err();
                assert!(err.contains_major(MajorErrorCode::Link), "{err:?}");
                assert!(err.contains_minor(MinorErrorCode::Exists), "{err:?}");
            }
            let file = File::open(&path).unwrap();
            let named = file.committed_datatype("mytype").unwrap();
            assert!(named.as_datatype().is_committed());
            assert_eq!(
                named.as_datatype().to_descriptor().unwrap(),
                dtype.to_descriptor().unwrap()
            );
            assert_eq!(named.name(), "/mytype");
            assert_eq!(file.committed_datatypes().unwrap().len(), 1);

            // names are relative to the group the method is called on
            let g = File::open_rw(&path).unwrap().create_group("g").unwrap();
            g.commit_datatype("t", &Datatype::from_type::<f64>().unwrap()).unwrap();
            assert!(g.committed_datatype("t").is_ok());
            assert!(g.committed_datatype("mytype").is_err());
            assert!(g.committed_datatype("/mytype").is_ok());
            assert!(g.file().unwrap().committed_datatype("g/t").is_ok());
        })
    }

    #[test]
    pub fn test_committed_datatype_attributes() {
        with_tmp_path(|path| {
            {
                let file = File::create(&path).unwrap();
                file.commit_datatype("mytype", &Datatype::from_type::<i32>().unwrap()).unwrap();
                let committed = file.committed_datatype("mytype").unwrap();
                committed.new_attr::<i32>().create("note").unwrap().write_scalar(&42).unwrap();
                committed.new_attr::<f64>().create("scale").unwrap().write_scalar(&0.5).unwrap();
            }
            let file = File::open(&path).unwrap();
            let committed = file.committed_datatype("mytype").unwrap();
            assert_eq!(committed.attr("note").unwrap().read_scalar::<i32>().unwrap(), 42);
            assert_eq!(committed.attr("scale").unwrap().read_scalar::<f64>().unwrap(), 0.5);
            let mut names = committed.attr_names().unwrap();
            names.sort();
            assert_eq!(names, ["note", "scale"]);
            assert_eq!(
                committed.as_datatype().to_descriptor().unwrap(),
                TypeDescriptor::Integer(IntSize::U4)
            );
        })
    }

    #[test]
    pub fn test_commit_datatype_missing_parent() {
        with_tmp_file(|file| {
            let dtype = Datatype::from_type::<i32>().unwrap();
            let err = file.commit_datatype("nogroup/t", &dtype).unwrap_err();
            assert!(err.contains_major(MajorErrorCode::SymbolTable), "{err:?}");
            assert!(err.contains_minor(MinorErrorCode::NotFound), "{err:?}");
            assert!(!dtype.is_committed());
            assert!(file.committed_datatypes().unwrap().is_empty());
        })
    }

    #[test]
    pub fn test_committed_datatypes_match_committed_datatype() {
        with_tmp_file(|file| {
            file.commit_datatype("ints", &Datatype::from_type::<i32>().unwrap()).unwrap();
            file.commit_datatype("strings", &Datatype::from_type::<VarLenUnicode>().unwrap())
                .unwrap();

            let mut listed: Vec<TypeDescriptor> = file
                .committed_datatypes()
                .unwrap()
                .iter()
                .map(|dt| dt.as_datatype().to_descriptor().unwrap())
                .collect();
            let mut opened: Vec<TypeDescriptor> = ["ints", "strings"]
                .iter()
                .map(|name| {
                    file.committed_datatype(name).unwrap().as_datatype().to_descriptor().unwrap()
                })
                .collect();
            listed.sort_by_key(|d| format!("{d:?}"));
            opened.sort_by_key(|d| format!("{d:?}"));
            assert_eq!(listed, opened);
        })
    }

    #[test]
    pub fn test_create_group_builder() {
        with_tmp_file(|file| {
            // the builder creates a named group
            let group = file.create_group_builder().create("foo").unwrap();
            assert_eq!(group.name(), "/foo");

            // intermediate groups are created by default, and can be disabled
            file.create_group_builder().create("a/b/c").unwrap();
            assert!(file.group("/a/b/c").is_ok());
            assert!(
                file.create_group_builder()
                    .create_intermediate_group(false)
                    .create("x/y/z")
                    .is_err()
            );
        })
    }

    #[test]
    pub fn test_create_group_anon() {
        with_tmp_file(|file| {
            // passing `None` creates an anonymous group
            let group = file.create_group_builder().create(None).unwrap();
            assert!(group.is_valid());
            // an anonymous group is not linked anywhere yet
            assert_eq!(file.len(), 0);
        })
    }

    #[test]
    pub fn test_group_track_times_default() {
        with_tmp_file(|file| {
            // groups track object times by default
            let group = file.create_group("default").unwrap();
            assert!(group.create_plist().unwrap().obj_track_times());
            // and the builder can request it explicitly
            let enabled =
                file.create_group_builder().obj_track_times(true).create("enabled").unwrap();
            assert!(enabled.create_plist().unwrap().obj_track_times());
        })
    }

    #[test]
    pub fn test_group_attr_creation_order() {
        with_tmp_file(|file| {
            let group = file
                .create_group_builder()
                .with_gcpl(|gcpl| {
                    gcpl.attr_creation_order(AttrCreationOrder::Indexed).attr_phase_change(2, 1)
                })
                .create("g")
                .unwrap();
            for name in ["c", "a", "b"] {
                group.new_attr::<u32>().create(name).unwrap();
            }

            let gcpl = group.gcpl().unwrap();
            assert_eq!(gcpl.attr_creation_order(), AttrCreationOrder::Indexed);
            assert_eq!(gcpl.attr_phase_change(), AttrPhaseChange { max_compact: 2, min_dense: 1 });
            assert_eq!(group.attr_names().unwrap(), ["a", "b", "c"]);

            let gcpl = file.create_group("untracked").unwrap().gcpl().unwrap();
            assert_eq!(gcpl.attr_creation_order(), AttrCreationOrder::Untracked);
            assert_eq!(gcpl.attr_phase_change(), AttrPhaseChange::default());
        })
    }

    // `obj_track_times` maps to a flag bit in the object header. Only a version-2
    // object header carries that flag (`H5O_HDR_STORE_TIMES` in the header prefix),
    // so only a v2 header can record the setting and report it back through
    // `H5Gget_create_plist`. A version-1 header has no such flag, so a disabled
    // setting is not stored and reads back as the default (enabled). Version-2
    // headers need at least the v18 file format. libhdf5 2.0 makes that the
    // default, older versions have to opt in with the library version bounds. The
    // `libver_*` API itself only exists from 1.10.2 on. See the format spec section
    // IV.A.1.b, where bit 5 of the version-2 prefix Flags field stores the times
    // (the version-1 prefix in IV.A.1.a has no Flags field):
    // https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t2.html#subsubsec_fmt2_dataobject_hdr_prefix_two
    #[cfg(feature = "1.10.2")]
    #[test]
    pub fn test_group_track_times_disabled() {
        // exercise both the minimum v18 format and the newest one
        for low in [LibraryVersion::V18, LibraryVersion::latest()] {
            with_tmp_path(|path| {
                let file = File::with_options()
                    .with_fapl(|fapl| fapl.libver_bounds(low, LibraryVersion::latest()))
                    .create(&path)
                    .unwrap();
                let group = file.create_group_builder().obj_track_times(false).create("g").unwrap();
                // the disabled setting round-trips through the created group
                assert!(!group.create_plist().unwrap().obj_track_times());
                // `gcpl` is an alias for `create_plist` and reports the same value
                assert!(!file.group("g").unwrap().gcpl().unwrap().obj_track_times());
            })
        }
    }

    #[cfg(feature = "1.10.2")]
    #[test]
    pub fn test_group_track_times_anon() {
        with_tmp_path(|path| {
            let file =
                File::with_options().with_fapl(|fapl| fapl.libver_v18()).create(&path).unwrap();
            // an anonymous group also honors the disabled time tracking setting
            let group = file.create_group_builder().obj_track_times(false).create(None).unwrap();
            assert!(group.is_valid());
            assert!(!group.create_plist().unwrap().obj_track_times());
            // and it is still not linked anywhere in the file
            assert_eq!(file.len(), 0);
        })
    }

    #[test]
    pub fn test_clone() {
        with_tmp_file(|file| {
            file.create_group("a").unwrap();
            let a = file.group("a").unwrap();
            assert_eq!(a.name(), "/a");
            assert_eq!(a.file().unwrap().id(), file.id());
            assert_eq!(a.refcount(), 1);
            let b = a.clone();
            assert_eq!(b.name(), "/a");
            assert_eq!(b.file().unwrap().id(), file.id());
            assert_eq!(b.refcount(), 2);
            assert_eq!(a.refcount(), 2);
            drop(a);
            assert_eq!(b.refcount(), 1);
            assert!(b.is_valid());
        })
    }

    #[test]
    pub fn test_group_info() {
        with_tmp_file(|file| {
            // A default group is a symbol table before 2.0 and a compact new-style group from 2.0
            let default_storage = if cfg!(feature = "2.0.0") {
                GroupStorageType::Compact
            } else {
                GroupStorageType::SymbolTable
            };
            let untracked = file.create_group("untracked").unwrap();
            let expected = GroupInfo {
                storage_type: default_storage,
                nlinks: 0,
                max_corder: 0,
                mounted: false,
            };
            assert_eq!(untracked.info().unwrap(), expected);
            untracked.create_group("a").unwrap();
            assert_eq!(untracked.info().unwrap(), GroupInfo { nlinks: 1, ..expected });

            let tracked = file
                .create_group_builder()
                .with_gcpl(|gcpl| gcpl.link_creation_order(LinkCreationOrder::Tracked))
                .create("tracked")
                .unwrap();
            for name in ["a", "b", "c"] {
                tracked.create_group(name).unwrap();
            }
            let expected = GroupInfo {
                storage_type: GroupStorageType::Compact,
                nlinks: 3,
                max_corder: 3,
                mounted: false,
            };
            assert_eq!(tracked.info().unwrap(), expected);

            // Unlinking leaves the creation order counter alone
            tracked.unlink("b").unwrap();
            assert_eq!(tracked.info().unwrap(), GroupInfo { nlinks: 2, ..expected });
            tracked.create_group("d").unwrap();
            assert_eq!(tracked.info().unwrap(), GroupInfo { nlinks: 3, max_corder: 4, ..expected });
            let d = tracked.find_link(IndexType::Name, IterationOrder::Increasing, |name, info| {
                if name == "d" { Ok(Some(info.creation_order)) } else { Ok(None) }
            });
            assert_eq!(d.unwrap(), Some(Some(3)));

            // More links than the compact limit move the group to dense storage
            for i in 0..8 {
                tracked.create_group(&format!("dense{i}")).unwrap();
            }
            let info = tracked.info().unwrap();
            assert_eq!(info.storage_type, GroupStorageType::Dense);
            assert_eq!((info.nlinks, info.max_corder), (11, 12));
        })
    }

    #[test]
    pub fn test_len() {
        with_tmp_file(|file| {
            assert_eq!(file.len(), 0);
            assert!(file.is_empty());
            file.create_group("foo").unwrap();
            assert_eq!(file.len(), 1);
            assert!(!file.is_empty());
            assert_eq!(file.group("foo").unwrap().len(), 0);
            assert!(file.group("foo").unwrap().is_empty());
            file.create_group("bar").unwrap().create_group("baz").unwrap();
            assert_eq!(file.len(), 2);
            assert_eq!(file.group("bar").unwrap().len(), 1);
            assert_eq!(file.group("/bar/baz").unwrap().len(), 0);
        })
    }

    #[test]
    pub fn test_link_hard() {
        with_tmp_file(|file| {
            file.create_group("foo/test/inner").unwrap();
            file.link_hard("/foo/test", "/foo/hard").unwrap();
            file.group("foo/test/inner").unwrap();
            file.group("/foo/hard/inner").unwrap();
            assert_err_re!(
                file.link_hard("foo/test", "/foo/test/inner"),
                "unable to (?:synchronously )?create (?:hard )?link: name already exists"
            );
            assert_err_re!(
                file.link_hard("foo/bar", "/foo/baz"),
                "unable to (?:synchronously )?create (?:hard )?link: object.+doesn't exist"
            );
            file.relink("/foo/hard", "/foo/hard2").unwrap();
            file.group("/foo/hard2/inner").unwrap();
            file.relink("/foo/test", "/foo/baz").unwrap();
            file.group("/foo/baz/inner").unwrap();
            file.group("/foo/hard2/inner").unwrap();
            file.unlink("/foo/baz").unwrap();
            assert_err_re!(file.group("/foo/baz"), "unable to (?:synchronously )?open group");
            file.group("/foo/hard2/inner").unwrap();
            file.unlink("/foo/hard2").unwrap();
            assert_err_re!(
                file.group("/foo/hard2/inner"),
                "unable to (?:synchronously )?open group"
            );
        })
    }

    #[test]
    pub fn test_link_soft() {
        with_tmp_file(|file| {
            file.create_group("a/b/c").unwrap();
            file.link_soft("/a/b", "a/soft").unwrap();
            file.group("/a/soft/c").unwrap();
            file.relink("/a/soft", "/a/soft2").unwrap();
            file.group("/a/soft2/c").unwrap();
            file.relink("a/b", "/a/d").unwrap();
            assert_err_re!(file.group("/a/soft2/c"), "unable to (?:synchronously )?open group");
            file.link_soft("/a/bar", "/a/baz").unwrap();
            assert_err_re!(file.group("/a/baz"), "unable to (?:synchronously )?open group");
            file.create_group("/a/bar").unwrap();
            file.group("/a/baz").unwrap();
            file.unlink("/a/bar").unwrap();
            assert_err_re!(file.group("/a/bar"), "unable to (?:synchronously )?open group");
            assert_err_re!(file.group("/a/baz"), "unable to (?:synchronously )?open group");
        })
    }

    #[test]
    pub fn test_link_exists() {
        with_tmp_file(|file| {
            file.create_group("a/b/c").unwrap();
            file.link_soft("/a/b", "a/soft").unwrap();
            file.group("/a/soft/c").unwrap();
            assert!(file.link_exists("a"));
            assert!(file.link_exists("a/b"));
            assert!(file.link_exists("a/b/c"));
            assert!(file.link_exists("a/soft"));
            assert!(file.link_exists("a/soft/c"));
            assert!(!file.link_exists("b"));
            assert!(!file.link_exists("soft"));
            let group = file.group("a/soft").unwrap();
            assert!(group.link_exists("c"));
            assert!(!group.link_exists("a"));
            assert!(!group.link_exists("soft"));
            #[cfg(not(feature = "1.10.0"))]
            assert!(!group.link_exists("/"));
            #[cfg(feature = "1.10.0")]
            assert!(group.link_exists("/"));
        })
    }

    #[test]
    pub fn test_relink() {
        with_tmp_file(|file| {
            file.create_group("test").unwrap();
            file.group("test").unwrap();
            assert_err!(
                file.relink("test", "foo/test"),
                "unable to move link: component not found"
            );
            file.create_group("foo").unwrap();
            assert_err!(file.relink("bar", "/baz"), "unable to move link: name doesn't exist");
            file.relink("test", "/foo/test").unwrap();
            file.group("/foo/test").unwrap();
            assert_err_re!(
                file.group("test"),
                "unable to (?:synchronously )?open group: object.+doesn't exist"
            );
        })
    }

    #[test]
    pub fn test_missing_group_error_codes() {
        with_tmp_file(|file| {
            file.create_group("a").unwrap();
            // Both a missing intermediate and a missing leaf report the same codes, so callers
            // can detect "no such object" without matching on the message text.
            for path in ["/foo/baz", "/a/baz"] {
                let err = file.group(path).unwrap_err();
                assert!(err.contains_major(MajorErrorCode::SymbolTable), "{path}: {err:?}");
                assert!(err.contains_minor(MinorErrorCode::NotFound), "{path}: {err:?}");
                assert!(!err.contains_minor(MinorErrorCode::NotHdf5), "{path}: {err:?}");
            }
            file.group("a").unwrap();
        })
    }

    #[test]
    pub fn test_unlink() {
        with_tmp_file(|file| {
            file.create_group("/foo/bar").unwrap();
            file.unlink("foo/bar").unwrap();
            assert_err_re!(file.group("/foo/bar"), "unable to (?:synchronously )?open group");
            assert!(file.group("foo").unwrap().is_empty());
        })
    }

    #[test]
    pub fn test_dataset() {
        with_tmp_file(|file| {
            file.new_dataset::<i32>().no_chunk().shape((10, 20)).create("/foo/bar").unwrap();
            file.new_dataset::<f32>()
                .shape(Extents::resizable((10, 20).into()))
                .create("baz")
                .unwrap();
            file.new_dataset::<u8>().shape((10.., 20..)).create(None).unwrap();
        });
    }

    #[test]
    pub fn test_get_member_names() {
        with_tmp_file(|file| {
            file.create_group("a").unwrap();
            file.create_group("b").unwrap();
            let group_a = file.group("a").unwrap();
            let group_b = file.group("b").unwrap();
            file.new_dataset::<u32>().no_chunk().shape((10, 20)).create("a/foo").unwrap();
            file.new_dataset::<u32>().no_chunk().shape((10, 20)).create("a/123").unwrap();
            file.new_dataset::<u32>().no_chunk().shape((10, 20)).create("a/bar").unwrap();
            let group_a_names = group_a.member_names().unwrap();
            assert!(group_a_names.contains(&"123".to_string()));
            assert!(group_a_names.contains(&"bar".to_string()));
            assert!(group_a_names.contains(&"foo".to_string()));
            assert_eq!(group_b.member_names().unwrap().len(), 0);
            let file_names = file.member_names().unwrap();
            assert!(file_names.contains(&"a".to_string()));
            assert!(file_names.contains(&"b".to_string()));
        })
    }

    #[test]
    pub fn test_external_link() {
        with_tmp_dir(|dir| {
            let file1 = dir.join("foo.h5");
            let file1 = File::create(file1).unwrap();
            let dset1 = file1.new_dataset::<i32>().create("foo").unwrap();
            dset1.write_scalar(&13).unwrap();

            let file2 = dir.join("bar.h5");
            let file2 = File::create(file2).unwrap();
            file2.link_external("foo.h5", "foo", "bar").unwrap();
            let dset2 = file2.dataset("bar").unwrap();
            assert_eq!(dset2.read_scalar::<i32>().unwrap(), 13);

            file1.unlink("foo").unwrap();
            assert!(file1.dataset("foo").is_err());
            assert!(file2.dataset("bar").is_err());

            // foo is only weakly closed
            assert_eq!(dset1.read_scalar::<i32>().unwrap(), 13);
            assert_eq!(dset2.read_scalar::<i32>().unwrap(), 13);
        })
    }

    #[test]
    pub fn test_iterators() {
        with_tmp_file(|file| {
            file.create_group("a").unwrap();
            file.create_group("b").unwrap();
            let group_a = file.group("a").unwrap();
            let _group_b = file.group("b").unwrap();
            file.new_dataset::<u32>().shape((10, 20)).create("a/foo").unwrap();
            file.new_dataset::<u32>().shape((10, 20)).create("a/123").unwrap();
            file.new_dataset::<u32>().shape((10, 20)).create("a/bar").unwrap();

            let groups = file.groups().unwrap();
            assert_eq!(groups.len(), 2);
            for group in groups {
                assert!(matches!(group.name().as_ref(), "/a" | "/b"));
            }

            let datasets = file.datasets().unwrap();
            assert_eq!(datasets.len(), 0);

            let datasets = group_a.datasets().unwrap();
            assert_eq!(datasets.len(), 3);
            for dataset in datasets {
                assert!(matches!(dataset.name().as_ref(), "/a/foo" | "/a/123" | "/a/bar"));
            }
        })
    }

    #[test]
    pub fn test_iter_visit_order() {
        with_tmp_file(|file| {
            let group = file.create_group("a").unwrap();
            for name in ["foo", "123", "bar"] {
                group.new_dataset::<u32>().create(name).unwrap();
            }
            let names = |order| group.member_names_by(IndexType::Name, order).unwrap();
            assert_eq!(names(IterationOrder::Increasing), ["123", "bar", "foo"]);
            assert_eq!(names(IterationOrder::Decreasing), ["foo", "bar", "123"]);

            let empty = file.create_group("empty").unwrap();
            assert!(
                empty.member_names_by(IndexType::Name, IterationOrder::Native).unwrap().is_empty()
            );
        })
    }

    #[test]
    pub fn test_iter_visit_creation_order() {
        with_tmp_file(|file| {
            let group = file
                .create_group_builder()
                .with_gcpl(|gcpl| gcpl.link_creation_order(LinkCreationOrder::Tracked))
                .create("a")
                .unwrap();
            for name in ["foo", "123", "bar"] {
                group.new_dataset::<u32>().create(name).unwrap();
            }

            let link = |name: &str, order| {
                let info = LinkInfo {
                    link_type: LinkType::Hard,
                    creation_order: Some(order),
                    char_encoding: CharEncoding::Ascii,
                };
                (name.to_owned(), info)
            };
            let foo = link("foo", 0);
            let num = link("123", 1);
            let bar = link("bar", 2);
            let links = |order| group.links(IndexType::CreationOrder, order).unwrap();
            assert_eq!(links(IterationOrder::Increasing), [foo.clone(), num.clone(), bar.clone()]);
            assert_eq!(links(IterationOrder::Decreasing), [bar, num, foo]);

            // A default group is a symbol table before 2.0 (BADVALUE, no creation order index)
            // and a new-style group from 2.0 (NOTFOUND, creation order not tracked)
            let not_tracked = if cfg!(feature = "2.0.0") {
                MinorErrorCode::NotFound
            } else {
                MinorErrorCode::BadValue
            };
            let untracked = file.create_group("b").unwrap();
            untracked.new_dataset::<u32>().create("foo").unwrap();
            let err = untracked
                .member_names_by(IndexType::CreationOrder, IterationOrder::Native)
                .unwrap_err();
            assert!(err.contains_major(MajorErrorCode::SymbolTable), "{err:?}");
            assert!(err.contains_minor(not_tracked), "{err:?}");
        })
    }

    #[test]
    pub fn test_find_link() {
        with_tmp_file(|file| {
            for name in ["a", "b", "c"] {
                file.create_group(name).unwrap();
            }

            let find = |wanted: &str| {
                let mut visited = vec![];
                let found = file
                    .find_link(IndexType::Name, IterationOrder::Increasing, |name, info| {
                        visited.push(name.to_owned());
                        if name == wanted {
                            Ok(Some((name.to_owned(), info.link_type)))
                        } else {
                            Ok(None)
                        }
                    })
                    .unwrap();
                (found, visited)
            };

            let (found, visited) = find("b");
            assert_eq!(found, Some(("b".to_owned(), LinkType::Hard)));
            assert_eq!(visited, ["a", "b"]);

            let (found, visited) = find("z");
            assert_eq!(found, None);
            assert_eq!(visited, ["a", "b", "c"]);
        })
    }

    #[test]
    pub fn test_iter_visit_error() {
        with_tmp_file(|file| {
            for name in ["a", "b", "c"] {
                file.create_group(name).unwrap();
            }

            let mut visited = vec![];
            let err = file
                .iter_visit(IndexType::Name, IterationOrder::Increasing, |name, _| {
                    visited.push(name.to_owned());
                    if name == "b" { Err("stop".into()) } else { Ok(()) }
                })
                .unwrap_err();
            assert_eq!(visited, ["a", "b"]);
            assert!(matches!(err, Error::Internal(ref msg) if msg == "stop"), "{err:?}");
            assert!(err.stack().is_none());
        })
    }

    #[test]
    pub fn test_iter_visit_from() {
        with_tmp_file(|file| {
            for name in ["a", "b", "c"] {
                file.create_group(name).unwrap();
            }
            let start = IterationCursor::start(IndexType::Name, IterationOrder::Increasing);

            let stop_at = |cursor, wanted: &str| {
                let mut visited = vec![];
                let stopped = file
                    .iter_visit_from(cursor, |name, _| {
                        visited.push(name.to_owned());
                        if name == wanted { Ok(Some(name.len())) } else { Ok(None) }
                    })
                    .unwrap();
                (stopped, visited)
            };

            let (stopped, visited) = stop_at(start, "b");
            assert_eq!(visited, ["a", "b"]);
            assert_eq!(stopped, Some((1, start.skip(2))));

            let (stopped, visited) = stop_at(start.skip(2), "c");
            assert_eq!(visited, ["c"]);
            assert_eq!(stopped, Some((1, start.skip(3))));

            let (stopped, visited) = stop_at(start.skip(2), "z");
            assert_eq!(visited, ["c"]);
            assert_eq!(stopped, None);

            for past_end in [start.skip(3), start.skip(9)] {
                let (stopped, visited) = stop_at(past_end, "a");
                assert!(visited.is_empty());
                assert_eq!(stopped, None);
            }
        })
    }

    #[test]
    pub fn test_iter_visit_panic() {
        with_tmp_file(|file| {
            file.create_group("a").unwrap();
            let payload = panic::catch_unwind(AssertUnwindSafe(|| {
                file.iter_visit_default(|_, _| -> Result<()> { panic!("boom") })
            }))
            .unwrap_err();
            assert_eq!(payload.downcast_ref::<&str>(), Some(&"boom"));

            let lock_is_free = std::thread::spawn(|| {
                crate::sync::LOCK.try_lock_for(std::time::Duration::from_secs(30)).is_some()
            });
            assert!(lock_is_free.join().unwrap());
            assert_eq!(file.member_names().unwrap(), ["a"]);
        })
    }

    #[test]
    pub fn test_iterators_unresolvable_link() {
        with_tmp_file(|file| {
            file.create_group("a").unwrap();
            file.link_soft("missing", "dangling").unwrap();

            assert_eq!(file.member_names().unwrap(), ["a", "dangling"]);
            let err = file.groups().unwrap_err();
            assert!(err.contains_major(MajorErrorCode::SymbolTable), "{err:?}");
            assert!(err.contains_minor(MinorErrorCode::NotFound), "{err:?}");
        })
    }
}
