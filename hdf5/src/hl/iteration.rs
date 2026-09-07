//! Traversal of the links of a group and the attributes of an object.

use std::any::Any;
use std::panic::{self, AssertUnwindSafe};
use std::ptr::addr_of_mut;

use hdf5_sys::h5::{H5_index_t, H5_iter_order_t, hsize_t};

use crate::internal_prelude::*;

/// The index the links of a group or the attributes of an object are traversed along.
///
/// Corresponds to `H5_index_t`. Traversing by [`CreationOrder`](Self::CreationOrder)
/// requires creation order to be tracked, see
/// [`LinkCreationOrder`](crate::plist::group_create::LinkCreationOrder) and
/// [`AttrCreationOrder`](crate::plist::group_create::AttrCreationOrder).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexType {
    /// Index on link names.
    Name,
    /// Index on link creation order.
    CreationOrder,
}

impl Default for IndexType {
    fn default() -> Self {
        Self::Name
    }
}

impl From<IndexType> for H5_index_t {
    fn from(v: IndexType) -> Self {
        match v {
            IndexType::Name => Self::H5_INDEX_NAME,
            IndexType::CreationOrder => Self::H5_INDEX_CRT_ORDER,
        }
    }
}

/// The order links or attributes are visited in along an [`IndexType`].
///
/// Corresponds to `H5_iter_order_t`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IterationOrder {
    /// Increasing order.
    Increasing,
    /// Decreasing order.
    Decreasing,
    /// No particular order, whatever is fastest.
    Native,
}

impl Default for IterationOrder {
    fn default() -> Self {
        Self::Native
    }
}

impl From<IterationOrder> for H5_iter_order_t {
    fn from(v: IterationOrder) -> Self {
        match v {
            IterationOrder::Increasing => Self::H5_ITER_INC,
            IterationOrder::Decreasing => Self::H5_ITER_DEC,
            IterationOrder::Native => Self::H5_ITER_NATIVE,
        }
    }
}

/// A position in a link or attribute iteration.
///
/// The cursor pairs the position with the [`IndexType`] and [`IterationOrder`] it
/// counts along, so an iteration can only be resumed the way it was started.
/// [`Group::iter_visit_from`](crate::Group::iter_visit_from) returns the cursor of a
/// stopped iteration and accepts it back to continue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IterationCursor {
    index_type: IndexType,
    iteration_order: IterationOrder,
    position: u64,
}

impl IterationCursor {
    /// Creates a cursor at the first link along `index_type` in `iteration_order`.
    pub const fn start(index_type: IndexType, iteration_order: IterationOrder) -> Self {
        Self { index_type, iteration_order, position: 0 }
    }

    /// Moves the cursor past the next `items` links or attributes.
    #[must_use]
    pub const fn skip(self, items: u64) -> Self {
        Self { position: self.position + items, ..self }
    }

    /// Returns the index type the cursor counts along.
    pub const fn index_type(self) -> IndexType {
        self.index_type
    }

    /// Returns the iteration order the cursor counts along.
    pub const fn iteration_order(self) -> IterationOrder {
        self.iteration_order
    }

    /// Returns the number of links or attributes before the cursor.
    pub const fn position(self) -> u64 {
        self.position
    }
}

pub(crate) type Callback<I> =
    unsafe extern "C" fn(hid_t, *const c_char, *const I, *mut c_void) -> herr_t;

/// Drives an HDF5 iteration from `cursor` until `op` returns a value.
///
/// `iterate` makes the C call with the index type, the order, the position, the
/// callback and its data. `count` is consulted only for a cursor past the start,
/// since HDF5 rejects a start position at or past the last item.
pub(crate) fn visit<I, T, B, F, N, C>(
    cursor: IterationCursor, count: N, op: F, iterate: C,
) -> Result<Option<(B, IterationCursor)>>
where
    T: for<'a> From<&'a I>,
    F: FnMut(&str, T) -> Result<Option<B>>,
    N: FnOnce() -> Result<u64>,
    C: FnOnce(H5_index_t, H5_iter_order_t, *mut hsize_t, Callback<I>, *mut c_void) -> herr_t,
{
    enum Stop<B> {
        Found(B),
        Error(Error),
        Panic(Box<dyn Any + Send>),
    }

    struct OpData<B, F> {
        op: F,
        stop: Option<Stop<B>>,
    }

    // Called by HDF5 once per item, never concurrently
    unsafe extern "C" fn callback<I, T, B, F>(
        _id: hid_t, name: *const c_char, info: *const I, op_data: *mut c_void,
    ) -> herr_t
    where
        T: for<'a> From<&'a I>,
        F: FnMut(&str, T) -> Result<Option<B>>,
    {
        // SAFETY: op_data is the pointer to the OpData passed to the C call below, which
        // outlives that call, and HDF5 does not run the callback concurrently
        let Some(data) = (unsafe { op_data.cast::<OpData<B, F>>().as_mut() }) else {
            return -1;
        };
        let visited = panic::catch_unwind(AssertUnwindSafe(|| {
            assert!(!name.is_null(), "iteration: null name ptr");
            // SAFETY: HDF5 passes a nul-terminated name that is valid for the duration of
            // the callback
            let name = unsafe { std::ffi::CStr::from_ptr(name) };
            // SAFETY: HDF5 passes a pointer to the item info that is valid for the duration
            // of the callback
            let info = unsafe { info.as_ref() }.expect("iteration: null info ptr");
            (data.op)(name.to_string_lossy().as_ref(), T::from(info))
        }));
        match visited {
            Ok(Ok(None)) => 0,
            Ok(Ok(Some(value))) => {
                data.stop = Some(Stop::Found(value));
                1
            }
            Ok(Err(err)) => {
                data.stop = Some(Stop::Error(err));
                -1
            }
            Err(payload) => {
                data.stop = Some(Stop::Panic(payload));
                -1
            }
        }
    }

    if cursor.position > 0 && cursor.position >= count()? {
        return Ok(None);
    }

    let mut data = OpData { op, stop: None };
    let mut position: hsize_t = cursor.position;
    let ret = h5call!(iterate(
        cursor.index_type.into(),
        cursor.iteration_order.into(),
        &mut position,
        callback::<I, T, B, F>,
        addr_of_mut!(data).cast::<c_void>()
    ));
    match data.stop {
        Some(Stop::Panic(payload)) => panic::resume_unwind(payload),
        Some(Stop::Error(err)) => Err(err),
        Some(Stop::Found(value)) => {
            ret?;
            Ok(Some((value, IterationCursor { position, ..cursor })))
        }
        None => {
            ret?;
            Ok(None)
        }
    }
}
