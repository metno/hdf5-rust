//! Attribute create properties.

use std::fmt::{self, Debug};
use std::ops::Deref;

use hdf5_sys::h5p::{H5Pcreate, H5Pget_char_encoding, H5Pset_char_encoding};
use hdf5_sys::h5t::{H5T_CSET_ASCII, H5T_CSET_UTF8, H5T_cset_t};

use crate::globals::H5P_ATTRIBUTE_CREATE;
pub use crate::hl::plist::link_create::CharEncoding;
use crate::internal_prelude::*;

/// Attribute create properties.
#[repr(transparent)]
pub struct AttributeCreate(Handle);

impl ObjectClass for AttributeCreate {
    const NAME: &'static str = "attribute create property list";
    const VALID_TYPES: &'static [H5I_type_t] = &[H5I_GENPROP_LST];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.is_class(PropertyListClass::AttributeCreate),
            "expected attribute create property list, got {:?}",
            self.class()
        );
        Ok(())
    }
}

impl Debug for AttributeCreate {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = f.debug_struct("AttributeCreate");
        formatter.field("char_encoding", &self.char_encoding());
        formatter.finish()
    }
}

impl Deref for AttributeCreate {
    type Target = PropertyList;

    fn deref(&self) -> &PropertyList {
        unsafe { self.transmute() }
    }
}

impl PartialEq for AttributeCreate {
    fn eq(&self, other: &Self) -> bool {
        <PropertyList as PartialEq>::eq(self, other)
    }
}

impl Eq for AttributeCreate {}

impl Clone for AttributeCreate {
    fn clone(&self) -> Self {
        unsafe { self.deref().clone().cast_unchecked() }
    }
}

/// Builder used to create attribute create property list.
#[derive(Clone, Debug, Default)]
pub struct AttributeCreateBuilder {
    char_encoding: Option<CharEncoding>,
}

impl AttributeCreateBuilder {
    /// Creates a new attribute create property list builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new builder from an existing property list.
    pub fn from_plist(plist: &AttributeCreate) -> Result<Self> {
        let mut builder = Self::default();
        builder.char_encoding(plist.get_char_encoding()?);
        Ok(builder)
    }

    /// Sets the character encoding of the attribute name.
    pub fn char_encoding(&mut self, encoding: CharEncoding) -> &mut Self {
        self.char_encoding = Some(encoding);
        self
    }

    fn populate_plist(&self, id: hid_t) -> Result<()> {
        if let Some(encoding) = self.char_encoding {
            let encoding = match encoding {
                CharEncoding::Ascii => H5T_CSET_ASCII,
                CharEncoding::Utf8 => H5T_CSET_UTF8,
            };
            h5try!(H5Pset_char_encoding(id, encoding));
        }
        Ok(())
    }

    /// Copies the builder settings into an attribute creation property list.
    pub fn apply(&self, plist: &mut AttributeCreate) -> Result<()> {
        h5lock!(self.populate_plist(plist.id()))
    }

    /// Constructs a new attribute creation property list.
    pub fn finish(&self) -> Result<AttributeCreate> {
        h5lock!({
            let mut plist = AttributeCreate::try_new()?;
            self.apply(&mut plist).map(|()| plist)
        })
    }
}

/// Attribute create property list.
impl AttributeCreate {
    /// Creates a new attribute creation property list.
    pub fn try_new() -> Result<Self> {
        Self::from_id(h5try!(H5Pcreate(*H5P_ATTRIBUTE_CREATE)))
    }

    /// Creates a copy of the attribute creation property list.
    pub fn copy(&self) -> Self {
        unsafe { self.deref().copy().cast_unchecked() }
    }

    /// Returns a builder for configuring an attribute creation property list.
    pub fn build() -> AttributeCreateBuilder {
        AttributeCreateBuilder::new()
    }

    #[doc(hidden)]
    pub fn get_char_encoding(&self) -> Result<CharEncoding> {
        Ok(match h5get!(H5Pget_char_encoding(self.id()): H5T_cset_t)? {
            H5T_CSET_ASCII => CharEncoding::Ascii,
            H5T_CSET_UTF8 => CharEncoding::Utf8,
            encoding => fail!("Unknown char encoding: {:?}", encoding),
        })
    }

    /// Returns the character encoding of the attribute name.
    pub fn char_encoding(&self) -> CharEncoding {
        self.get_char_encoding().unwrap_or(CharEncoding::Ascii)
    }
}
