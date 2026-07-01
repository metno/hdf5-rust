//! Group creation properties.

use std::fmt::{self, Debug};
use std::ops::Deref;

use hdf5_sys::h5p::{H5Pcreate, H5Pget_obj_track_times, H5Pset_obj_track_times};

use crate::globals::H5P_GROUP_CREATE;
use crate::internal_prelude::*;

/// Group creation properties.
#[repr(transparent)]
#[derive(Clone)]
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

impl Debug for GroupCreate {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = f.debug_struct("GroupCreate");
        formatter.field("obj_track_times", &self.obj_track_times());
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
        Ok(builder)
    }

    /// Sets a property that governs the recording of times associated with an object.
    ///
    /// If true, time data will be recorded; if false, time data will not be recorded.
    pub fn obj_track_times(&mut self, track_times: bool) -> &mut Self {
        self.obj_track_times = Some(track_times);
        self
    }

    fn populate_plist(&self, id: hid_t) -> Result<()> {
        if let Some(v) = self.obj_track_times {
            h5try!(H5Pset_obj_track_times(id, hbool_t::from(v)));
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
}
