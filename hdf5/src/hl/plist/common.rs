use std::os::raw::c_uint;

use hdf5_sys::h5p::{H5P_CRT_ORDER_INDEXED, H5P_CRT_ORDER_TRACKED};

use bitflags::bitflags;

/// Attribute storage phase change thresholds.
///
/// These thresholds determine the point at which attribute storage changes from
/// compact storage (i.e., storage in the object header) to dense storage (i.e.,
/// storage in a heap and indexed with a B-tree).
///
/// In the general case, attributes are initially kept in compact storage. When
/// the number of attributes exceeds `max_compact`, attribute storage switches to
/// dense storage. If the number of attributes subsequently falls below `min_dense`,
/// the attributes are returned to compact storage.
///
/// If `max_compact` is set to 0 (zero), dense storage always used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttrPhaseChange {
    /// Maximum number of attributes to be stored in compact storage (default: 8).
    pub max_compact: u32,
    /// Minimum number of attributes to be stored in dense storage (default: 6).
    pub min_dense: u32,
}

impl Default for AttrPhaseChange {
    fn default() -> Self {
        Self { max_compact: 8, min_dense: 6 }
    }
}

bitflags! {
    /// Flags for tracking and indexing attribute creation order of an object.
    ///
    /// Default behavior is that attribute creation order is neither tracked nor indexed.
    ///
    /// Note that if a creation order index is to be built, it must be specified in
    /// the object creation property list. HDF5 currently provides no mechanism to turn
    /// on attribute creation order tracking at object creation time and to build the
    /// index later.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
    pub struct AttrCreationOrder: u32 {
        /// Attribute creation order is tracked but not necessarily indexed.
        const TRACKED = H5P_CRT_ORDER_TRACKED as _;
        /// Attribute creation order is indexed (requires to be tracked).
        const INDEXED = H5P_CRT_ORDER_INDEXED as _;
    }
}

/// Tracking of link creation order in a group.
///
/// By default link creation order is not recorded. `Tracked` records the order in
/// which links are created and allows the group to be traversed by
/// [`IndexType::CreationOrder`](crate::IndexType::CreationOrder). `Indexed` also
/// maintains an index for that traversal.
///
/// The setting is fixed in the creation property list. HDF5 provides no way to
/// turn on tracking or build the index after the group exists.
///
/// # Examples
///
/// ```
/// use hdf5_metno::plist::GroupCreateBuilder;
/// use hdf5_metno::plist::group_create::LinkCreationOrder;
///
/// let gcpl = GroupCreateBuilder::new().link_creation_order(LinkCreationOrder::Indexed).finish()?;
/// assert_eq!(gcpl.link_creation_order(), LinkCreationOrder::Indexed);
/// # Ok::<(), hdf5_metno::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LinkCreationOrder {
    /// Link creation order is not recorded.
    #[default]
    Untracked,
    /// Link creation order is recorded.
    Tracked,
    /// Link creation order is recorded and indexed.
    Indexed,
}

impl LinkCreationOrder {
    pub(crate) fn from_flags(flags: c_uint) -> Self {
        if flags & H5P_CRT_ORDER_INDEXED != 0 {
            Self::Indexed
        } else if flags & H5P_CRT_ORDER_TRACKED != 0 {
            Self::Tracked
        } else {
            Self::Untracked
        }
    }
}

impl From<LinkCreationOrder> for c_uint {
    fn from(v: LinkCreationOrder) -> Self {
        match v {
            LinkCreationOrder::Untracked => 0,
            LinkCreationOrder::Tracked => H5P_CRT_ORDER_TRACKED,
            LinkCreationOrder::Indexed => H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED,
        }
    }
}
