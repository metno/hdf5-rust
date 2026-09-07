//! Group creation properties.

use std::fmt::{self, Debug};
use std::ops::Deref;

use hdf5_sys::h5p::{
    H5Pcreate, H5Pget_attr_creation_order, H5Pget_attr_phase_change, H5Pget_link_creation_order,
    H5Pget_obj_track_times, H5Pset_attr_creation_order, H5Pset_attr_phase_change,
    H5Pset_link_creation_order, H5Pset_obj_track_times,
};

use crate::globals::H5P_GROUP_CREATE;
pub use crate::hl::plist::common::{AttrCreationOrder, AttrPhaseChange, LinkCreationOrder};
use crate::internal_prelude::*;

/// Group creation properties.
#[repr(transparent)]
pub struct GroupCreate(Handle);

impl ObjectClass for GroupCreate {
    const NAME: &'static str = "group create property list";
    const VALID_TYPES: &'static [H5I_type_t] = &[H5I_GENPROP_LST];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.is_class(PropertyListClass::GroupCreate),
            "expected group create property list, got {:?}",
            self.class()
        );
        Ok(())
    }
}

impl Clone for GroupCreate {
    fn clone(&self) -> Self {
        unsafe { self.deref().clone().cast_unchecked() }
    }
}

impl Debug for GroupCreate {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = f.debug_struct("GroupCreate");
        formatter.field("obj_track_times", &self.obj_track_times());
        formatter.field("link_creation_order", &self.link_creation_order());
        formatter.field("attr_phase_change", &self.attr_phase_change());
        formatter.field("attr_creation_order", &self.attr_creation_order());
        formatter.finish()
    }
}

impl Deref for GroupCreate {
    type Target = PropertyList;

    fn deref(&self) -> &PropertyList {
        unsafe { self.transmute() }
    }
}

impl PartialEq for GroupCreate {
    fn eq(&self, other: &Self) -> bool {
        <PropertyList as PartialEq>::eq(self, other)
    }
}

impl Eq for GroupCreate {}

/// Builder used to create a group creation property list.
#[derive(Clone, Debug, Default)]
pub struct GroupCreateBuilder {
    obj_track_times: Option<bool>,
    link_creation_order: Option<LinkCreationOrder>,
    attr_phase_change: Option<AttrPhaseChange>,
    attr_creation_order: Option<AttrCreationOrder>,
}

impl GroupCreateBuilder {
    /// Creates a new group creation property list builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new builder from an existing property list.
    pub fn from_plist(plist: &GroupCreate) -> Result<Self> {
        let mut builder = Self::default();
        builder.obj_track_times(plist.get_obj_track_times()?);
        builder.link_creation_order(plist.get_link_creation_order()?);
        let apc = plist.get_attr_phase_change()?;
        builder.attr_phase_change(apc.max_compact, apc.min_dense);
        builder.attr_creation_order(plist.get_attr_creation_order()?);
        Ok(builder)
    }

    /// Sets a property that governs the recording of times associated with an object.
    ///
    /// If true, time data will be recorded; if false, time data will not be recorded.
    pub fn obj_track_times(&mut self, track_times: bool) -> &mut Self {
        self.obj_track_times = Some(track_times);
        self
    }

    /// Sets whether link creation order is tracked and indexed.
    ///
    /// See [`LinkCreationOrder`] for the available settings.
    pub fn link_creation_order(&mut self, link_creation_order: LinkCreationOrder) -> &mut Self {
        self.link_creation_order = Some(link_creation_order);
        self
    }

    /// Sets the group's attribute storage phase change thresholds.
    ///
    /// See [`AttrPhaseChange`] for the meaning of the thresholds.
    pub fn attr_phase_change(&mut self, max_compact: u32, min_dense: u32) -> &mut Self {
        self.attr_phase_change = Some(AttrPhaseChange { max_compact, min_dense });
        self
    }

    /// Sets whether the group's attribute creation order is tracked and indexed.
    ///
    /// See [`AttrCreationOrder`] for the available settings.
    pub fn attr_creation_order(&mut self, attr_creation_order: AttrCreationOrder) -> &mut Self {
        self.attr_creation_order = Some(attr_creation_order);
        self
    }

    fn populate_plist(&self, id: hid_t) -> Result<()> {
        if let Some(v) = self.obj_track_times {
            h5try!(H5Pset_obj_track_times(id, hbool_t::from(v)));
        }
        if let Some(v) = self.link_creation_order {
            h5try!(H5Pset_link_creation_order(id, v.into()));
        }
        if let Some(v) = self.attr_phase_change {
            h5try!(H5Pset_attr_phase_change(id, v.max_compact as _, v.min_dense as _));
        }
        if let Some(v) = self.attr_creation_order {
            h5try!(H5Pset_attr_creation_order(id, v.into()));
        }
        Ok(())
    }

    /// Copies the builder settings into a group creation property list.
    pub fn apply(&self, plist: &mut GroupCreate) -> Result<()> {
        h5lock!(self.populate_plist(plist.id()))
    }

    /// Constructs a new group creation property list.
    pub fn finish(&self) -> Result<GroupCreate> {
        h5lock!({
            let mut plist = GroupCreate::try_new()?;
            self.apply(&mut plist).map(|()| plist)
        })
    }
}

/// Group creation property list.
impl GroupCreate {
    /// Creates a new group creation property list.
    pub fn try_new() -> Result<Self> {
        Self::from_id(h5try!(H5Pcreate(*H5P_GROUP_CREATE)))
    }

    /// Creates a copy of the group creation property list.
    pub fn copy(&self) -> Self {
        unsafe { self.deref().copy().cast_unchecked() }
    }

    /// Returns a builder for configuring a group creation property list.
    pub fn build() -> GroupCreateBuilder {
        GroupCreateBuilder::new()
    }

    #[doc(hidden)]
    pub fn get_obj_track_times(&self) -> Result<bool> {
        h5get!(H5Pget_obj_track_times(self.id()): hbool_t).map(|x| x > 0)
    }

    /// Returns true if the time data is recorded.
    pub fn obj_track_times(&self) -> bool {
        self.get_obj_track_times().unwrap_or(true)
    }

    #[doc(hidden)]
    pub fn get_link_creation_order(&self) -> Result<LinkCreationOrder> {
        h5get!(H5Pget_link_creation_order(self.id()): c_uint).map(LinkCreationOrder::from_flags)
    }

    /// Returns whether link creation order is tracked and indexed.
    ///
    /// Returns [`LinkCreationOrder::Untracked`] if the property cannot be read.
    pub fn link_creation_order(&self) -> LinkCreationOrder {
        self.get_link_creation_order().unwrap_or_default()
    }

    #[doc(hidden)]
    pub fn get_attr_phase_change(&self) -> Result<AttrPhaseChange> {
        h5get!(H5Pget_attr_phase_change(self.id()): c_uint, c_uint)
            .map(|(mc, md)| AttrPhaseChange { max_compact: mc as _, min_dense: md as _ })
    }

    /// Returns the group's attribute storage phase change thresholds.
    ///
    /// Returns the HDF5 defaults if the property cannot be read.
    pub fn attr_phase_change(&self) -> AttrPhaseChange {
        self.get_attr_phase_change().unwrap_or_default()
    }

    #[doc(hidden)]
    pub fn get_attr_creation_order(&self) -> Result<AttrCreationOrder> {
        h5get!(H5Pget_attr_creation_order(self.id()): c_uint).map(AttrCreationOrder::from_flags)
    }

    /// Returns whether the group's attribute creation order is tracked and indexed.
    ///
    /// Returns [`AttrCreationOrder::Untracked`] if the property cannot be read.
    pub fn attr_creation_order(&self) -> AttrCreationOrder {
        self.get_attr_creation_order().unwrap_or_default()
    }
}
