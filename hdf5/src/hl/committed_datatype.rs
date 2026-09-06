use std::fmt::{self, Debug};
use std::ops::Deref;

use hdf5_sys::h5t::H5Tcommitted;

use crate::internal_prelude::*;

/// Represents a committed HDF5 datatype: a datatype stored in a file as an object of its own,
/// which can carry attributes like a group or a dataset.
#[repr(transparent)]
pub struct CommittedDatatype(Handle);

impl ObjectClass for CommittedDatatype {
    const NAME: &'static str = "committed datatype";
    const VALID_TYPES: &'static [H5I_type_t] = &[H5I_DATATYPE];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            h5call!(H5Tcommitted(self.id())).unwrap_or(0) > 0,
            "expected committed datatype, got a transient datatype"
        );
        Ok(())
    }

    fn short_repr(&self) -> Option<String> {
        self.as_datatype().short_repr()
    }
}

impl Debug for CommittedDatatype {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.debug_fmt(f)
    }
}

impl Clone for CommittedDatatype {
    fn clone(&self) -> Self {
        unsafe { self.deref().clone().cast_unchecked() }
    }
}

impl Deref for CommittedDatatype {
    type Target = Location;

    fn deref(&self) -> &Location {
        // SAFETY: both types are `#[repr(transparent)]` over `Handle`, and `H5I_DATATYPE` is one
        // of the id types `Location` accepts.
        unsafe { self.transmute() }
    }
}

impl TryFrom<Datatype> for CommittedDatatype {
    type Error = Error;

    /// Converts a datatype into a committed datatype, failing for a transient datatype.
    fn try_from(datatype: Datatype) -> Result<Self> {
        // SAFETY: both types are `#[repr(transparent)]` over `Handle` and accept `H5I_DATATYPE` ids,
        // and `validate` checks that the datatype is committed before the value is returned.
        let committed: Self = unsafe { datatype.cast_unchecked() };
        committed.validate().map(|()| committed)
    }
}

impl CommittedDatatype {
    /// Returns the datatype stored in this object.
    pub fn as_datatype(&self) -> &Datatype {
        // SAFETY: both types are `#[repr(transparent)]` over `Handle` and accept `H5I_DATATYPE` ids.
        unsafe { self.transmute() }
    }
}

#[cfg(test)]
pub mod tests {
    use crate::internal_prelude::*;

    #[test]
    pub fn test_committed_datatype_conversion() {
        with_tmp_file(|file| {
            let transient = Datatype::from_type::<i32>().unwrap();
            assert_err!(CommittedDatatype::try_from(transient), "expected committed datatype");

            let dtype = Datatype::from_type::<i32>().unwrap();
            file.commit_datatype("mytype", &dtype).unwrap();
            let committed = CommittedDatatype::try_from(dtype).unwrap();
            assert_eq!(format!("{committed:?}"), "<HDF5 committed datatype: int32>");
            assert_eq!(committed.name(), "/mytype");
            assert!(committed.as_datatype().is_committed());
        })
    }
}
