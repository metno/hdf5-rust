//! Object copy properties.
use std::fmt::{self, Debug};
use std::ops::Deref;

use hdf5_sys::h5o::{
    H5O_COPY_EXPAND_EXT_LINK_FLAG, H5O_COPY_EXPAND_SOFT_LINK_FLAG, H5O_COPY_SHALLOW_HIERARCHY_FLAG,
    H5O_COPY_WITHOUT_ATTR_FLAG,
};
use hdf5_sys::h5p::{H5Pcreate, H5Pget_copy_object, H5Pset_copy_object};

use crate::globals::H5P_OBJECT_COPY;
use crate::internal_prelude::*;

/// Object copy properties.
#[repr(transparent)]
pub struct ObjectCopy(Handle);

impl ObjectClass for ObjectCopy {
    const NAME: &'static str = "object copy property list";
    const VALID_TYPES: &'static [H5I_type_t] = &[H5I_GENPROP_LST];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.is_class(PropertyListClass::ObjectCopy),
            "expected object copy property list, got {:?}",
            self.class()
        );
        Ok(())
    }
}

impl Debug for ObjectCopy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = f.debug_struct("ObjectCopy");
        formatter.field("copy_without_attr", &self.copy_without_attr());
        formatter.field("shallow_hierarchy", &self.shallow_hierarchy());
        formatter.field("expand_soft_links", &self.expand_soft_links());
        formatter.field("expand_ext_links", &self.expand_ext_links());
        formatter.finish()
    }
}

impl Deref for ObjectCopy {
    type Target = PropertyList;

    fn deref(&self) -> &PropertyList {
        unsafe { self.transmute() }
    }
}

impl PartialEq for ObjectCopy {
    fn eq(&self, other: &Self) -> bool {
        <PropertyList as PartialEq>::eq(self, other)
    }
}

impl Eq for ObjectCopy {}

impl Clone for ObjectCopy {
    fn clone(&self) -> Self {
        unsafe { self.deref().clone().cast_unchecked() }
    }
}

/// Builder used to create object copy property list.
#[derive(Clone, Debug, Default)]
pub struct ObjectCopyBuilder {
    copy_without_attr: Option<bool>,
    shallow_hierarchy: Option<bool>,
    expand_soft_links: Option<bool>,
    expand_ext_links: Option<bool>,
}

impl ObjectCopyBuilder {
    /// Creates a new object copy property list builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new builder from an existing property list.
    pub fn from_plist(plist: &ObjectCopy) -> Result<Self> {
        let mut builder = Self::default();
        let flags = plist.get_flags()?;
        builder.copy_without_attr = Some(flags & H5O_COPY_WITHOUT_ATTR_FLAG != 0);
        builder.shallow_hierarchy = Some(flags & H5O_COPY_SHALLOW_HIERARCHY_FLAG != 0);
        builder.expand_soft_links = Some(flags & H5O_COPY_EXPAND_SOFT_LINK_FLAG != 0);
        builder.expand_ext_links = Some(flags & H5O_COPY_EXPAND_EXT_LINK_FLAG != 0);
        Ok(builder)
    }

    /// Copy object without copying attributes.
    pub fn copy_without_attr(&mut self, enable: bool) -> &mut Self {
        self.copy_without_attr = Some(enable);
        self
    }

    /// Copy only immediate members of a group (shallow copy).
    pub fn shallow_hierarchy(&mut self, enable: bool) -> &mut Self {
        self.shallow_hierarchy = Some(enable);
        self
    }

    /// Expand soft links into new objects.
    pub fn expand_soft_links(&mut self, enable: bool) -> &mut Self {
        self.expand_soft_links = Some(enable);
        self
    }

    /// Expand external links into new objects.
    pub fn expand_ext_links(&mut self, enable: bool) -> &mut Self {
        self.expand_ext_links = Some(enable);
        self
    }

    fn populate_plist(&self, id: hid_t) -> Result<()> {
        let mut flags = 0u32;

        if self.copy_without_attr.unwrap_or(false) {
            flags |= H5O_COPY_WITHOUT_ATTR_FLAG;
        }
        if self.shallow_hierarchy.unwrap_or(false) {
            flags |= H5O_COPY_SHALLOW_HIERARCHY_FLAG;
        }
        if self.expand_soft_links.unwrap_or(false) {
            flags |= H5O_COPY_EXPAND_SOFT_LINK_FLAG;
        }
        if self.expand_ext_links.unwrap_or(false) {
            flags |= H5O_COPY_EXPAND_EXT_LINK_FLAG;
        }

        if flags != 0 {
            h5try!(H5Pset_copy_object(id, flags));
        }

        Ok(())
    }

    pub fn apply(&self, plist: &mut ObjectCopy) -> Result<()> {
        h5lock!(self.populate_plist(plist.id()))
    }

    pub fn finish(&self) -> Result<ObjectCopy> {
        h5lock!({
            let mut plist = ObjectCopy::try_new()?;
            self.apply(&mut plist).map(|()| plist)
        })
    }
}

/// Object copy property list.
impl ObjectCopy {
    pub fn try_new() -> Result<Self> {
        Self::from_id(h5try!(H5Pcreate(*H5P_OBJECT_COPY)))
    }

    pub fn copy(&self) -> Self {
        unsafe { self.deref().copy().cast_unchecked() }
    }

    pub fn build() -> ObjectCopyBuilder {
        ObjectCopyBuilder::new()
    }

    #[doc(hidden)]
    pub fn get_flags(&self) -> Result<c_uint> {
        h5get!(H5Pget_copy_object(self.id()): c_uint)
    }

    pub fn copy_without_attr(&self) -> bool {
        self.get_flags().map(|f| f & H5O_COPY_WITHOUT_ATTR_FLAG != 0).unwrap_or(false)
    }

    pub fn shallow_hierarchy(&self) -> bool {
        self.get_flags().map(|f| f & H5O_COPY_SHALLOW_HIERARCHY_FLAG != 0).unwrap_or(false)
    }

    pub fn expand_soft_links(&self) -> bool {
        self.get_flags().map(|f| f & H5O_COPY_EXPAND_SOFT_LINK_FLAG != 0).unwrap_or(false)
    }

    pub fn expand_ext_links(&self) -> bool {
        self.get_flags().map(|f| f & H5O_COPY_EXPAND_EXT_LINK_FLAG != 0).unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_object_copy_builder_from_plist() {
        let ocpypl =
            ObjectCopy::build().copy_without_attr(true).expand_soft_links(true).finish().unwrap();

        let builder = ObjectCopyBuilder::from_plist(&ocpypl).unwrap();
        let ocpypl2 = builder.finish().unwrap();

        assert_eq!(ocpypl.copy_without_attr(), ocpypl2.copy_without_attr());
        assert_eq!(ocpypl.shallow_hierarchy(), ocpypl2.shallow_hierarchy());
        assert_eq!(ocpypl.expand_soft_links(), ocpypl2.expand_soft_links());
        assert_eq!(ocpypl.expand_ext_links(), ocpypl2.expand_ext_links());
        assert_eq!(ocpypl, ocpypl2);
    }
}
