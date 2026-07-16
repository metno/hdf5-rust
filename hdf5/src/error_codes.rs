//! Major and minor error codes reported by the HDF5 library.
//!
//! Each [`ErrorFrame`](crate::ErrorFrame) carries a major code for which part of the library
//! failed and a minor code for how it failed. HDF5 assigns these ids at runtime, so this module
//! resolves them into enums that can be matched.
//!
//! Variants are not gated on the HDF5 version, so naming one never requires a `#[cfg]`. A code
//! the linked HDF5 does not define is never returned by `from_id`, and a code this crate does
//! not know becomes `Other`.
//!
//! Codes HDF5 renamed (but kept for backwards compatibility) map to one variant, so a match works on any
//! version: [`MinorErrorCode::BadId`] resolves from `H5E_BADID` or `H5E_BADATOM`, and
//! [`MinorErrorCode::Logging`] from `H5E_LOGGING` or `H5E_LOGFAIL`.
//!
//! Names, descriptions and version gates come from HDF5's [`H5err.txt`], which generates the
//! `H5E_*_g` symbols. To check a gate, diff that file between two release tags, e.g.
//! `hdf5-1_10_2` (no `H5E_CONTEXT`) against `hdf5-1_10_3` (has it). The [user guide] describes
//! major and minor codes.
//!
//! [`H5err.txt`]: https://github.com/HDFGroup/hdf5/blob/hdf5_2.1.0/src/H5err.txt
//! [user guide]: https://support.hdfgroup.org/documentation/hdf5/latest/_h5_e__u_g.html
//!
//! # Example
//!
//! ```no_run
//! use hdf5_metno as hdf5;
//! use hdf5::MinorErrorCode;
//!
//! # let dir = tempfile::tempdir().unwrap();
//! # let path = dir.path().join("corrupt.h5");
//! # std::fs::write(&path, b"definitely not an HDF5 file").unwrap();
//! match hdf5::File::open(&path) {
//!     Err(err) if err.contains_minor(MinorErrorCode::NotHdf5) => {
//!         // the file exists but its superblock is unreadable
//!     }
//!     _ => {}
//! }
//! ```

use std::collections::HashMap;
use std::fmt::{self, Display};
use std::sync::LazyLock;

use hdf5_sys::h5i::hid_t;

use crate::globals::*;

/// Defines an error-code enum from three lists:
///
/// - `variants`: the enum body, never gated (no need for `#[cfg]` downstream).
/// - `symbols`: `hid_t` to variant, gated on the version introducing the symbol. Renamed codes
///   contribute one entry per spelling.
/// - `meta`: variant to C identifier and description. Gated only for renamed codes, where the
///   arms must be mutually exclusive and total.
macro_rules! error_codes {
    (
        $(#[$emeta:meta])*
        $name:ident, $table:ident;

        variants { $( $(#[$vdoc:meta])* $variant:ident, )* }
        symbols { $( $([$scfg:meta])? $global:ident => $svariant:ident, )* }
        meta { $( $([$mcfg:meta])? $mvariant:ident => ($cname:literal, $desc:literal), )* }
    ) => {
        $(#[$emeta])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
        #[non_exhaustive]
        pub enum $name {
            $(
                $(#[$vdoc])*
                $variant,
            )*
            /// A code reported by HDF5 that this crate does not know.
            ///
            /// The raw message id is only valid for the lifetime of the process.
            Other(hid_t),
        }

        // Initialised from inside `h5lock`, which is only safe because `sync` forces
        // `LIBRARY_INIT` before taking `LOCK`, so dereferencing the globals below cannot
        // reach for the lock a second time.
        pub(crate) static $table: LazyLock<HashMap<hid_t, $name>> = LazyLock::new(|| {
            let mut map = HashMap::new();
            $(
                $(#[cfg($scfg)])?
                map.insert(*$global, $name::$svariant);
            )*
            map
        });

        impl $name {
            /// Resolves a raw HDF5 message id, or [`Other`](Self::Other) if unknown.
            #[must_use]
            pub fn from_id(id: hid_t) -> Self {
                $table.get(&id).copied().unwrap_or(Self::Other(id))
            }

            /// The C identifier, e.g. `"H5E_CANTOPENFILE"`, as spelled by the linked HDF5.
            ///
            /// `None` only for [`Other`](Self::Other).
            #[must_use]
            pub fn name(self) -> Option<&'static str> {
                match self {
                    $( $(#[cfg($mcfg)])? Self::$mvariant => Some($cname), )*
                    Self::Other(_) => None,
                }
            }

            /// The description HDF5 documents for this code, as worded by the linked HDF5.
            ///
            /// `None` only for [`Other`](Self::Other).
            #[must_use]
            pub fn description(self) -> Option<&'static str> {
                match self {
                    $( $(#[cfg($mcfg)])? Self::$mvariant => Some($desc), )*
                    Self::Other(_) => None,
                }
            }

            /// Every code this crate knows, including any the linked HDF5 cannot report.
            #[must_use]
            pub fn all() -> &'static [Self] {
                &[ $( Self::$variant, )* ]
            }
        }

        impl Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                match *self {
                    Self::Other(id) => write!(f, "unknown error code ({id})"),
                    ref code => f.write_str(code.description().unwrap_or("unknown error code")),
                }
            }
        }
    };
}

error_codes! {
    /// The major error code of an HDF5 error frame.
    MajorErrorCode, MAJOR_CODES;

    variants {
        /// `H5E_ARGS`: Invalid arguments to routine
        Args,
        /// `H5E_ATTR`: Attribute
        Attr,
        /// `H5E_BTREE`: B-Tree node
        BTree,
        /// `H5E_CACHE`: Object cache
        Cache,
        /// `H5E_CONTEXT`: API Context
        Context,
        /// `H5E_DATASET`: Dataset
        Dataset,
        /// `H5E_DATASPACE`: Dataspace
        Dataspace,
        /// `H5E_DATATYPE`: Datatype
        Datatype,
        /// `H5E_ERROR`: Error API
        ErrorApi,
        /// `H5E_EVENTSET`: Event Set
        EventSet,
        /// `H5E_EARRAY`: Extensible Array
        ExtensibleArray,
        /// `H5E_EFL`: External file list
        ExternalFileList,
        /// `H5E_FILE`: File accessibility
        File,
        /// `H5E_FARRAY`: Fixed Array
        FixedArray,
        /// `H5E_FSPACE`: Free Space Manager
        FreeSpace,
        /// `H5E_FUNC`: Function entry/exit
        Func,
        /// `H5E_HEAP`: Heap
        Heap,
        /// `H5E_ID` (`H5E_ATOM` in older HDF5): Object ID
        Id,
        /// `H5E_INTERNAL`: Internal error (too specific to document in detail)
        Internal,
        /// `H5E_IO`: Low-level I/O
        Io,
        /// `H5E_LIB`: General library infrastructure
        Lib,
        /// `H5E_LINK`: Links
        Link,
        /// `H5E_MAP`: Map
        Map,
        /// `H5E_NONE_MAJOR`: No error
        NoneMajor,
        /// `H5E_OHDR`: Object header
        ObjectHeader,
        /// `H5E_PAGEBUF`: Page Buffering
        PageBuffering,
        /// `H5E_PLINE`: Data filters
        Pipeline,
        /// `H5E_PLUGIN`: Plugin for dynamically loaded library
        Plugin,
        /// `H5E_PLIST`: Property lists
        PropertyList,
        /// `H5E_RTREE`: R-Tree spatial index
        RTree,
        /// `H5E_RS`: Reference Counted Strings
        RefCountedString,
        /// `H5E_REFERENCE`: References
        Reference,
        /// `H5E_RESOURCE`: Resource unavailable
        Resource,
        /// `H5E_SOHM`: Shared Object Header Messages
        SharedObjectHeaderMessage,
        /// `H5E_SLIST`: Skip Lists
        SkipList,
        /// `H5E_STORAGE`: Data storage
        Storage,
        /// `H5E_SYM`: Symbol table
        SymbolTable,
        /// `H5E_TST`: Ternary Search Trees
        TernarySearchTree,
        /// `H5E_THREADSAFE`: Threadsafety
        Threadsafety,
        /// `H5E_VFL`: Virtual File Layer
        VirtualFileLayer,
        /// `H5E_VOL`: Virtual Object Layer
        VirtualObjectLayer,
    }

    symbols {
        H5E_ARGS => Args,
        H5E_ATTR => Attr,
        H5E_BTREE => BTree,
        H5E_CACHE => Cache,
        [feature = "1.10.3"] H5E_CONTEXT => Context,
        H5E_DATASET => Dataset,
        H5E_DATASPACE => Dataspace,
        H5E_DATATYPE => Datatype,
        H5E_ERROR => ErrorApi,
        [feature = "1.14.0"] H5E_EVENTSET => EventSet,
        [feature = "1.10.0"] H5E_EARRAY => ExtensibleArray,
        H5E_EFL => ExternalFileList,
        H5E_FILE => File,
        [feature = "1.10.0"] H5E_FARRAY => FixedArray,
        H5E_FSPACE => FreeSpace,
        H5E_FUNC => Func,
        H5E_HEAP => Heap,
        [not(feature = "1.14.0")] H5E_ATOM => Id,
        [feature = "1.14.0"] H5E_ID => Id,
        H5E_INTERNAL => Internal,
        H5E_IO => Io,
        [feature = "1.12.1"] H5E_LIB => Lib,
        H5E_LINK => Link,
        [feature = "1.12.0"] H5E_MAP => Map,
        H5E_NONE_MAJOR => NoneMajor,
        H5E_OHDR => ObjectHeader,
        [feature = "1.10.1"] H5E_PAGEBUF => PageBuffering,
        H5E_PLINE => Pipeline,
        [feature = "1.8.11"] H5E_PLUGIN => Plugin,
        H5E_PLIST => PropertyList,
        [feature = "2.0.0"] H5E_RTREE => RTree,
        H5E_RS => RefCountedString,
        H5E_REFERENCE => Reference,
        H5E_RESOURCE => Resource,
        H5E_SOHM => SharedObjectHeaderMessage,
        H5E_SLIST => SkipList,
        H5E_STORAGE => Storage,
        H5E_SYM => SymbolTable,
        H5E_TST => TernarySearchTree,
        [feature = "2.0.0"] H5E_THREADSAFE => Threadsafety,
        H5E_VFL => VirtualFileLayer,
        [feature = "1.12.0"] H5E_VOL => VirtualObjectLayer,
    }

    meta {
        Args => ("H5E_ARGS", "Invalid arguments to routine"),
        Attr => ("H5E_ATTR", "Attribute"),
        BTree => ("H5E_BTREE", "B-Tree node"),
        Cache => ("H5E_CACHE", "Object cache"),
        Context => ("H5E_CONTEXT", "API Context"),
        Dataset => ("H5E_DATASET", "Dataset"),
        Dataspace => ("H5E_DATASPACE", "Dataspace"),
        Datatype => ("H5E_DATATYPE", "Datatype"),
        ErrorApi => ("H5E_ERROR", "Error API"),
        EventSet => ("H5E_EVENTSET", "Event Set"),
        ExtensibleArray => ("H5E_EARRAY", "Extensible Array"),
        ExternalFileList => ("H5E_EFL", "External file list"),
        File => ("H5E_FILE", "File accessibility"),
        FixedArray => ("H5E_FARRAY", "Fixed Array"),
        FreeSpace => ("H5E_FSPACE", "Free Space Manager"),
        Func => ("H5E_FUNC", "Function entry/exit"),
        Heap => ("H5E_HEAP", "Heap"),
        [feature = "1.14.0"] Id => ("H5E_ID", "Object ID"),
        [not(feature = "1.14.0")] Id => ("H5E_ATOM", "Object atom"),
        Internal => ("H5E_INTERNAL", "Internal error (too specific to document in detail)"),
        Io => ("H5E_IO", "Low-level I/O"),
        Lib => ("H5E_LIB", "General library infrastructure"),
        Link => ("H5E_LINK", "Links"),
        Map => ("H5E_MAP", "Map"),
        NoneMajor => ("H5E_NONE_MAJOR", "No error"),
        ObjectHeader => ("H5E_OHDR", "Object header"),
        PageBuffering => ("H5E_PAGEBUF", "Page Buffering"),
        Pipeline => ("H5E_PLINE", "Data filters"),
        Plugin => ("H5E_PLUGIN", "Plugin for dynamically loaded library"),
        PropertyList => ("H5E_PLIST", "Property lists"),
        RTree => ("H5E_RTREE", "R-Tree spatial index"),
        RefCountedString => ("H5E_RS", "Reference Counted Strings"),
        Reference => ("H5E_REFERENCE", "References"),
        Resource => ("H5E_RESOURCE", "Resource unavailable"),
        SharedObjectHeaderMessage => ("H5E_SOHM", "Shared Object Header Messages"),
        SkipList => ("H5E_SLIST", "Skip Lists"),
        Storage => ("H5E_STORAGE", "Data storage"),
        SymbolTable => ("H5E_SYM", "Symbol table"),
        TernarySearchTree => ("H5E_TST", "Ternary Search Trees"),
        Threadsafety => ("H5E_THREADSAFE", "Threadsafety"),
        VirtualFileLayer => ("H5E_VFL", "Virtual File Layer"),
        VirtualObjectLayer => ("H5E_VOL", "Virtual Object Layer"),
    }
}

error_codes! {
    /// The minor error code of an HDF5 error frame.
    MinorErrorCode, MINOR_CODES;

    variants {
        /// `H5E_ALIGNMENT`: Alignment error
        Alignment,
        /// `H5E_ALREADYEXISTS`: Object already exists
        AlreadyExists,
        /// `H5E_ALREADYINIT`: Object already initialized
        AlreadyInit,
        /// `H5E_BADFILE`: Bad file ID accessed
        BadFile,
        /// `H5E_BADGROUP`: Unable to find ID group information
        BadGroup,
        /// `H5E_BADID` (`H5E_BADATOM` in older HDF5): Unable to find ID information (already closed?)
        BadId,
        /// `H5E_BADITER`: Iteration failed
        BadIter,
        /// `H5E_BADMESG`: Unrecognized message
        BadMessage,
        /// `H5E_BADRANGE`: Out of range
        BadRange,
        /// `H5E_BADSELECT`: Invalid selection
        BadSelect,
        /// `H5E_BADSIZE`: Bad size for object
        BadSize,
        /// `H5E_BADTYPE`: Inappropriate type
        BadType,
        /// `H5E_BADVALUE`: Bad value
        BadValue,
        /// `H5E_CALLBACK`: Callback failed
        Callback,
        /// `H5E_CANAPPLY`: Error from filter 'can apply' callback
        CanApply,
        /// `H5E_CANTALLOC`: Can't allocate space
        CantAlloc,
        /// `H5E_CANTAPPEND`: Can't append object
        CantAppend,
        /// `H5E_CANTATTACH`: Can't attach object
        CantAttach,
        /// `H5E_CANTCANCEL`: Can't cancel operation
        CantCancel,
        /// `H5E_CANTCLEAN`: Unable to mark metadata as clean
        CantClean,
        /// `H5E_CANTCLIP`: Can't clip hyperslab region
        CantClip,
        /// `H5E_CANTCLOSEFILE`: Unable to close file
        CantCloseFile,
        /// `H5E_CANTCLOSEOBJ`: Can't close object
        CantCloseObj,
        /// `H5E_CANTCOMPARE`: Can't compare objects
        CantCompare,
        /// `H5E_CANTCOMPUTE`: Can't compute value
        CantCompute,
        /// `H5E_CANTCONVERT`: Can't convert datatypes
        CantConvert,
        /// `H5E_CANTCOPY`: Unable to copy object
        CantCopy,
        /// `H5E_CANTCORK`: Unable to cork an object
        CantCork,
        /// `H5E_CANTCOUNT`: Can't count elements
        CantCount,
        /// `H5E_CANTCREATE`: Unable to create file
        CantCreate,
        /// `H5E_CANTDEC`: Unable to decrement reference count
        CantDec,
        /// `H5E_CANTDECODE`: Unable to decode value
        CantDecode,
        /// `H5E_CANTDELETE`: Can't delete message
        CantDelete,
        /// `H5E_CANTDELETEFILE`: Unable to delete file
        CantDeleteFile,
        /// `H5E_CANTDEPEND`: Unable to create a flush dependency
        CantDepend,
        /// `H5E_CANTDIRTY`: Unable to mark metadata as dirty
        CantDirty,
        /// `H5E_CANTENCODE`: Unable to encode value
        CantEncode,
        /// `H5E_CANTEXPUNGE`: Unable to expunge a metadata cache entry
        CantExpunge,
        /// `H5E_CANTEXTEND`: Can't extend heap's space
        CantExtend,
        /// `H5E_CANTFILTER`: Filter operation failed
        CantFilter,
        /// `H5E_CANTFIND`: Unable to check for record
        CantFind,
        /// `H5E_CANTFLUSH`: Unable to flush data from cache
        CantFlush,
        /// `H5E_CANTFREE`: Unable to free object
        CantFree,
        /// `H5E_CANTGATHER`: Can't gather data
        CantGather,
        /// `H5E_CANTGC`: Unable to garbage collect
        CantGc,
        /// `H5E_CANTGET`: Can't get value
        CantGet,
        /// `H5E_CANTGETSIZE`: Unable to compute size
        CantGetSize,
        /// `H5E_CANTINC`: Unable to increment reference count
        CantInc,
        /// `H5E_CANTINIT`: Unable to initialize object
        CantInit,
        /// `H5E_CANTINS`: Unable to insert metadata into cache
        CantIns,
        /// `H5E_CANTINSERT`: Unable to insert object
        CantInsert,
        /// `H5E_CANTLIST`: Unable to list node
        CantList,
        /// `H5E_CANTLOAD`: Unable to load metadata into cache
        CantLoad,
        /// `H5E_CANTLOCK`: Unable to lock object
        CantLock,
        /// `H5E_CANTLOCKFILE`: Unable to lock file
        CantLockFile,
        /// `H5E_CANTMARKCLEAN`: Unable to mark a pinned entry as clean
        CantMarkClean,
        /// `H5E_CANTMARKDIRTY`: Unable to mark a pinned entry as dirty
        CantMarkDirty,
        /// `H5E_CANTMARKSERIALIZED`: Unable to mark an entry as serialized
        CantMarkSerialized,
        /// `H5E_CANTMARKUNSERIALIZED`: Unable to mark an entry as unserialized
        CantMarkUnserialized,
        /// `H5E_CANTMERGE`: Can't merge objects
        CantMerge,
        /// `H5E_CANTMODIFY`: Unable to modify record
        CantModify,
        /// `H5E_CANTMOVE`: Can't move object
        CantMove,
        /// `H5E_CANTNEXT`: Can't move to next iterator location
        CantNext,
        /// `H5E_CANTNOTIFY`: Unable to notify object about action
        CantNotify,
        /// `H5E_CANTOPENFILE`: Unable to open file
        CantOpenFile,
        /// `H5E_CANTOPENOBJ`: Can't open object
        CantOpenObj,
        /// `H5E_CANTOPERATE`: Can't operate on object
        CantOperate,
        /// `H5E_CANTPACK`: Can't pack messages
        CantPack,
        /// `H5E_CANTPIN`: Unable to pin cache entry
        CantPin,
        /// `H5E_CANTPROTECT`: Unable to protect metadata
        CantProtect,
        /// `H5E_CANTPUT`: Can't put value
        CantPut,
        /// `H5E_CANTRECV`: Can't receive data
        CantRecv,
        /// `H5E_CANTREDISTRIBUTE`: Unable to redistribute records
        CantRedistribute,
        /// `H5E_CANTREGISTER`: Unable to register new ID
        CantRegister,
        /// `H5E_CANTRELEASE`: Unable to release object
        CantRelease,
        /// `H5E_CANTREMOVE`: Unable to remove object
        CantRemove,
        /// `H5E_CANTRENAME`: Unable to rename object
        CantRename,
        /// `H5E_CANTRESET`: Can't reset object
        CantReset,
        /// `H5E_CANTRESIZE`: Unable to resize a metadata cache entry
        CantResize,
        /// `H5E_CANTRESTORE`: Can't restore condition
        CantRestore,
        /// `H5E_CANTREVIVE`: Can't revive object
        CantRevive,
        /// `H5E_CANTSELECT`: Can't select hyperslab
        CantSelect,
        /// `H5E_CANTSERIALIZE`: Unable to serialize data from cache
        CantSerialize,
        /// `H5E_CANTSET`: Can't set value
        CantSet,
        /// `H5E_CANTSHRINK`: Can't shrink container
        CantShrink,
        /// `H5E_CANTSORT`: Can't sort objects
        CantSort,
        /// `H5E_CANTSPLIT`: Unable to split node
        CantSplit,
        /// `H5E_CANTSWAP`: Unable to swap records
        CantSwap,
        /// `H5E_CANTTAG`: Unable to tag metadata in the cache
        CantTag,
        /// `H5E_CANTUNCORK`: Unable to uncork an object
        CantUncork,
        /// `H5E_CANTUNDEPEND`: Unable to destroy a flush dependency
        CantUndepend,
        /// `H5E_CANTUNLOCK`: Unable to unlock object
        CantUnlock,
        /// `H5E_CANTUNLOCKFILE`: Unable to unlock file
        CantUnlockFile,
        /// `H5E_CANTUNPIN`: Unable to un-pin cache entry
        CantUnpin,
        /// `H5E_CANTUNPROTECT`: Unable to unprotect metadata
        CantUnprotect,
        /// `H5E_CANTUNSERIALIZE`: Unable to mark metadata as unserialized
        CantUnserialize,
        /// `H5E_CANTUPDATE`: Can't update object
        CantUpdate,
        /// `H5E_CANTWAIT`: Can't wait on operation
        CantWait,
        /// `H5E_CLOSEERROR`: Close failed
        CloseError,
        /// `H5E_COMPLEN`: Name component is too long
        CompLen,
        /// `H5E_DUPCLASS`: Duplicate class name in parent class
        DupClass,
        /// `H5E_EXISTS`: Object already exists
        Exists,
        /// `H5E_FCNTL`: File control (fcntl) failed
        Fcntl,
        /// `H5E_FILEEXISTS`: File already exists
        FileExists,
        /// `H5E_FILEOPEN`: File already open
        FileOpen,
        /// `H5E_INCONSISTENTSTATE`: Internal states are inconsistent
        InconsistentState,
        /// `H5E_LINKCOUNT`: Bad object header link count
        LinkCount,
        /// `H5E_LOGGING` (`H5E_LOGFAIL` in older HDF5): Failure in the cache logging framework
        Logging,
        /// `H5E_MOUNT`: File mount error
        Mount,
        /// `H5E_MPI`: Some MPI function failed
        Mpi,
        /// `H5E_MPIERRSTR`: MPI Error String
        MpiErrStr,
        /// `H5E_NLINKS`: Too many soft links in path
        NLinks,
        /// `H5E_NOENCODER`: Filter present but encoding disabled
        NoEncoder,
        /// `H5E_NOFILTER`: Requested filter is not available
        NoFilter,
        /// `H5E_NOIDS`: Out of IDs for group
        NoIds,
        /// `H5E_NO_INDEPENDENT`: Can't perform independent IO
        NoIndependent,
        /// `H5E_NOSPACE`: No space available for allocation
        NoSpace,
        /// `H5E_NONE_MINOR`: No error
        NoneMinor,
        /// `H5E_NOTCACHED`: Metadata not currently cached
        NotCached,
        /// `H5E_NOTFOUND`: Object not found
        NotFound,
        /// `H5E_NOTHDF5`: Not an HDF5 file
        NotHdf5,
        /// `H5E_NOTREGISTERED`: Link class not registered
        NotRegistered,
        /// `H5E_OBJOPEN`: Object is already open
        ObjOpen,
        /// `H5E_OPENERROR`: Can't open directory or file
        OpenError,
        /// `H5E_OVERFLOW`: Address overflowed
        Overflow,
        /// `H5E_PATH`: Problem with path to object
        Path,
        /// `H5E_PROTECT`: Protected metadata error
        Protect,
        /// `H5E_READERROR`: Read failed
        ReadError,
        /// `H5E_SEEKERROR`: Seek failed
        SeekError,
        /// `H5E_SETDISALLOWED`: Disallowed operation
        SetDisallowed,
        /// `H5E_SETLOCAL`: Error from filter 'set local' callback
        SetLocal,
        /// `H5E_SYSERRSTR`: System error message
        SysErrStr,
        /// `H5E_SYSTEM`: Internal error detected
        System,
        /// `H5E_TRAVERSE`: Link traversal failure
        Traverse,
        /// `H5E_TRUNCATED`: File has been truncated
        Truncated,
        /// `H5E_UNINITIALIZED`: Information is uinitialized
        Uninitialized,
        /// `H5E_UNMOUNT`: File unmount error
        Unmount,
        /// `H5E_UNSUPPORTED`: Feature is unsupported
        Unsupported,
        /// `H5E_VERSION`: Wrong version number
        Version,
        /// `H5E_WRITEERROR`: Write failed
        WriteError,
    }

    symbols {
        H5E_ALIGNMENT => Alignment,
        H5E_ALREADYEXISTS => AlreadyExists,
        H5E_ALREADYINIT => AlreadyInit,
        H5E_BADFILE => BadFile,
        H5E_BADGROUP => BadGroup,
        [not(feature = "1.14.0")] H5E_BADATOM => BadId,
        [feature = "1.14.0"] H5E_BADID => BadId,
        H5E_BADITER => BadIter,
        H5E_BADMESG => BadMessage,
        H5E_BADRANGE => BadRange,
        H5E_BADSELECT => BadSelect,
        H5E_BADSIZE => BadSize,
        H5E_BADTYPE => BadType,
        H5E_BADVALUE => BadValue,
        H5E_CALLBACK => Callback,
        H5E_CANAPPLY => CanApply,
        H5E_CANTALLOC => CantAlloc,
        [feature = "1.10.0"] H5E_CANTAPPEND => CantAppend,
        H5E_CANTATTACH => CantAttach,
        [feature = "1.14.0"] H5E_CANTCANCEL => CantCancel,
        [feature = "1.10.1"] H5E_CANTCLEAN => CantClean,
        H5E_CANTCLIP => CantClip,
        H5E_CANTCLOSEFILE => CantCloseFile,
        H5E_CANTCLOSEOBJ => CantCloseObj,
        H5E_CANTCOMPARE => CantCompare,
        H5E_CANTCOMPUTE => CantCompute,
        H5E_CANTCONVERT => CantConvert,
        H5E_CANTCOPY => CantCopy,
        [feature = "1.10.0"] H5E_CANTCORK => CantCork,
        H5E_CANTCOUNT => CantCount,
        H5E_CANTCREATE => CantCreate,
        H5E_CANTDEC => CantDec,
        H5E_CANTDECODE => CantDecode,
        H5E_CANTDELETE => CantDelete,
        [feature = "1.12.0"] H5E_CANTDELETEFILE => CantDeleteFile,
        [feature = "1.10.0"] H5E_CANTDEPEND => CantDepend,
        H5E_CANTDIRTY => CantDirty,
        H5E_CANTENCODE => CantEncode,
        H5E_CANTEXPUNGE => CantExpunge,
        H5E_CANTEXTEND => CantExtend,
        H5E_CANTFILTER => CantFilter,
        [feature = "1.14.0"] H5E_CANTFIND => CantFind,
        H5E_CANTFLUSH => CantFlush,
        H5E_CANTFREE => CantFree,
        [feature = "1.10.2"] H5E_CANTGATHER => CantGather,
        H5E_CANTGC => CantGc,
        H5E_CANTGET => CantGet,
        H5E_CANTGETSIZE => CantGetSize,
        H5E_CANTINC => CantInc,
        H5E_CANTINIT => CantInit,
        H5E_CANTINS => CantIns,
        H5E_CANTINSERT => CantInsert,
        H5E_CANTLIST => CantList,
        H5E_CANTLOAD => CantLoad,
        H5E_CANTLOCK => CantLock,
        [any(all(feature = "1.10.7", not(feature = "1.12.0")), feature = "1.12.1")] H5E_CANTLOCKFILE => CantLockFile,
        [feature = "1.10.1"] H5E_CANTMARKCLEAN => CantMarkClean,
        H5E_CANTMARKDIRTY => CantMarkDirty,
        [feature = "1.10.1"] H5E_CANTMARKSERIALIZED => CantMarkSerialized,
        [feature = "1.10.1"] H5E_CANTMARKUNSERIALIZED => CantMarkUnserialized,
        H5E_CANTMERGE => CantMerge,
        H5E_CANTMODIFY => CantModify,
        H5E_CANTMOVE => CantMove,
        H5E_CANTNEXT => CantNext,
        [feature = "1.10.0"] H5E_CANTNOTIFY => CantNotify,
        H5E_CANTOPENFILE => CantOpenFile,
        H5E_CANTOPENOBJ => CantOpenObj,
        H5E_CANTOPERATE => CantOperate,
        H5E_CANTPACK => CantPack,
        H5E_CANTPIN => CantPin,
        H5E_CANTPROTECT => CantProtect,
        [feature = "1.14.0"] H5E_CANTPUT => CantPut,
        H5E_CANTRECV => CantRecv,
        H5E_CANTREDISTRIBUTE => CantRedistribute,
        H5E_CANTREGISTER => CantRegister,
        H5E_CANTRELEASE => CantRelease,
        H5E_CANTREMOVE => CantRemove,
        H5E_CANTRENAME => CantRename,
        H5E_CANTRESET => CantReset,
        H5E_CANTRESIZE => CantResize,
        H5E_CANTRESTORE => CantRestore,
        H5E_CANTREVIVE => CantRevive,
        H5E_CANTSELECT => CantSelect,
        H5E_CANTSERIALIZE => CantSerialize,
        H5E_CANTSET => CantSet,
        H5E_CANTSHRINK => CantShrink,
        H5E_CANTSORT => CantSort,
        H5E_CANTSPLIT => CantSplit,
        H5E_CANTSWAP => CantSwap,
        [feature = "1.10.0"] H5E_CANTTAG => CantTag,
        [feature = "1.10.0"] H5E_CANTUNCORK => CantUncork,
        [feature = "1.10.0"] H5E_CANTUNDEPEND => CantUndepend,
        H5E_CANTUNLOCK => CantUnlock,
        [any(all(feature = "1.10.7", not(feature = "1.12.0")), feature = "1.12.1")] H5E_CANTUNLOCKFILE => CantUnlockFile,
        H5E_CANTUNPIN => CantUnpin,
        H5E_CANTUNPROTECT => CantUnprotect,
        [feature = "1.10.1"] H5E_CANTUNSERIALIZE => CantUnserialize,
        H5E_CANTUPDATE => CantUpdate,
        [feature = "1.14.0"] H5E_CANTWAIT => CantWait,
        H5E_CLOSEERROR => CloseError,
        H5E_COMPLEN => CompLen,
        H5E_DUPCLASS => DupClass,
        H5E_EXISTS => Exists,
        H5E_FCNTL => Fcntl,
        H5E_FILEEXISTS => FileExists,
        H5E_FILEOPEN => FileOpen,
        [feature = "1.10.7"] H5E_INCONSISTENTSTATE => InconsistentState,
        H5E_LINKCOUNT => LinkCount,
        [all(feature = "1.10.0", not(feature = "1.12.0"))] H5E_LOGFAIL => Logging,
        [feature = "1.10.5"] H5E_LOGGING => Logging,
        H5E_MOUNT => Mount,
        H5E_MPI => Mpi,
        H5E_MPIERRSTR => MpiErrStr,
        H5E_NLINKS => NLinks,
        H5E_NOENCODER => NoEncoder,
        H5E_NOFILTER => NoFilter,
        H5E_NOIDS => NoIds,
        [feature = "1.10.2"] H5E_NO_INDEPENDENT => NoIndependent,
        H5E_NOSPACE => NoSpace,
        H5E_NONE_MINOR => NoneMinor,
        H5E_NOTCACHED => NotCached,
        H5E_NOTFOUND => NotFound,
        H5E_NOTHDF5 => NotHdf5,
        H5E_NOTREGISTERED => NotRegistered,
        H5E_OBJOPEN => ObjOpen,
        [feature = "1.8.11"] H5E_OPENERROR => OpenError,
        H5E_OVERFLOW => Overflow,
        H5E_PATH => Path,
        H5E_PROTECT => Protect,
        H5E_READERROR => ReadError,
        H5E_SEEKERROR => SeekError,
        [feature = "1.8.9"] H5E_SETDISALLOWED => SetDisallowed,
        H5E_SETLOCAL => SetLocal,
        H5E_SYSERRSTR => SysErrStr,
        H5E_SYSTEM => System,
        H5E_TRAVERSE => Traverse,
        H5E_TRUNCATED => Truncated,
        H5E_UNINITIALIZED => Uninitialized,
        [feature = "1.14.0"] H5E_UNMOUNT => Unmount,
        H5E_UNSUPPORTED => Unsupported,
        H5E_VERSION => Version,
        H5E_WRITEERROR => WriteError,
    }

    meta {
        Alignment => ("H5E_ALIGNMENT", "Alignment error"),
        AlreadyExists => ("H5E_ALREADYEXISTS", "Object already exists"),
        AlreadyInit => ("H5E_ALREADYINIT", "Object already initialized"),
        BadFile => ("H5E_BADFILE", "Bad file ID accessed"),
        BadGroup => ("H5E_BADGROUP", "Unable to find ID group information"),
        [feature = "1.14.0"] BadId => ("H5E_BADID", "Unable to find ID information (already closed?)"),
        [not(feature = "1.14.0")] BadId => ("H5E_BADATOM", "Unable to find atom information (already closed?)"),
        BadIter => ("H5E_BADITER", "Iteration failed"),
        BadMessage => ("H5E_BADMESG", "Unrecognized message"),
        BadRange => ("H5E_BADRANGE", "Out of range"),
        BadSelect => ("H5E_BADSELECT", "Invalid selection"),
        BadSize => ("H5E_BADSIZE", "Bad size for object"),
        BadType => ("H5E_BADTYPE", "Inappropriate type"),
        BadValue => ("H5E_BADVALUE", "Bad value"),
        Callback => ("H5E_CALLBACK", "Callback failed"),
        CanApply => ("H5E_CANAPPLY", "Error from filter 'can apply' callback"),
        CantAlloc => ("H5E_CANTALLOC", "Can't allocate space"),
        CantAppend => ("H5E_CANTAPPEND", "Can't append object"),
        CantAttach => ("H5E_CANTATTACH", "Can't attach object"),
        CantCancel => ("H5E_CANTCANCEL", "Can't cancel operation"),
        CantClean => ("H5E_CANTCLEAN", "Unable to mark metadata as clean"),
        CantClip => ("H5E_CANTCLIP", "Can't clip hyperslab region"),
        CantCloseFile => ("H5E_CANTCLOSEFILE", "Unable to close file"),
        CantCloseObj => ("H5E_CANTCLOSEOBJ", "Can't close object"),
        CantCompare => ("H5E_CANTCOMPARE", "Can't compare objects"),
        CantCompute => ("H5E_CANTCOMPUTE", "Can't compute value"),
        CantConvert => ("H5E_CANTCONVERT", "Can't convert datatypes"),
        CantCopy => ("H5E_CANTCOPY", "Unable to copy object"),
        CantCork => ("H5E_CANTCORK", "Unable to cork an object"),
        CantCount => ("H5E_CANTCOUNT", "Can't count elements"),
        CantCreate => ("H5E_CANTCREATE", "Unable to create file"),
        CantDec => ("H5E_CANTDEC", "Unable to decrement reference count"),
        CantDecode => ("H5E_CANTDECODE", "Unable to decode value"),
        CantDelete => ("H5E_CANTDELETE", "Can't delete message"),
        CantDeleteFile => ("H5E_CANTDELETEFILE", "Unable to delete file"),
        CantDepend => ("H5E_CANTDEPEND", "Unable to create a flush dependency"),
        CantDirty => ("H5E_CANTDIRTY", "Unable to mark metadata as dirty"),
        CantEncode => ("H5E_CANTENCODE", "Unable to encode value"),
        CantExpunge => ("H5E_CANTEXPUNGE", "Unable to expunge a metadata cache entry"),
        CantExtend => ("H5E_CANTEXTEND", "Can't extend heap's space"),
        CantFilter => ("H5E_CANTFILTER", "Filter operation failed"),
        CantFind => ("H5E_CANTFIND", "Unable to check for record"),
        CantFlush => ("H5E_CANTFLUSH", "Unable to flush data from cache"),
        CantFree => ("H5E_CANTFREE", "Unable to free object"),
        CantGather => ("H5E_CANTGATHER", "Can't gather data"),
        CantGc => ("H5E_CANTGC", "Unable to garbage collect"),
        CantGet => ("H5E_CANTGET", "Can't get value"),
        CantGetSize => ("H5E_CANTGETSIZE", "Unable to compute size"),
        CantInc => ("H5E_CANTINC", "Unable to increment reference count"),
        CantInit => ("H5E_CANTINIT", "Unable to initialize object"),
        CantIns => ("H5E_CANTINS", "Unable to insert metadata into cache"),
        CantInsert => ("H5E_CANTINSERT", "Unable to insert object"),
        CantList => ("H5E_CANTLIST", "Unable to list node"),
        CantLoad => ("H5E_CANTLOAD", "Unable to load metadata into cache"),
        CantLock => ("H5E_CANTLOCK", "Unable to lock object"),
        CantLockFile => ("H5E_CANTLOCKFILE", "Unable to lock file"),
        CantMarkClean => ("H5E_CANTMARKCLEAN", "Unable to mark a pinned entry as clean"),
        CantMarkDirty => ("H5E_CANTMARKDIRTY", "Unable to mark a pinned entry as dirty"),
        CantMarkSerialized => ("H5E_CANTMARKSERIALIZED", "Unable to mark an entry as serialized"),
        CantMarkUnserialized => ("H5E_CANTMARKUNSERIALIZED", "Unable to mark an entry as unserialized"),
        CantMerge => ("H5E_CANTMERGE", "Can't merge objects"),
        CantModify => ("H5E_CANTMODIFY", "Unable to modify record"),
        CantMove => ("H5E_CANTMOVE", "Can't move object"),
        CantNext => ("H5E_CANTNEXT", "Can't move to next iterator location"),
        CantNotify => ("H5E_CANTNOTIFY", "Unable to notify object about action"),
        CantOpenFile => ("H5E_CANTOPENFILE", "Unable to open file"),
        CantOpenObj => ("H5E_CANTOPENOBJ", "Can't open object"),
        CantOperate => ("H5E_CANTOPERATE", "Can't operate on object"),
        CantPack => ("H5E_CANTPACK", "Can't pack messages"),
        CantPin => ("H5E_CANTPIN", "Unable to pin cache entry"),
        CantProtect => ("H5E_CANTPROTECT", "Unable to protect metadata"),
        CantPut => ("H5E_CANTPUT", "Can't put value"),
        CantRecv => ("H5E_CANTRECV", "Can't receive data"),
        CantRedistribute => ("H5E_CANTREDISTRIBUTE", "Unable to redistribute records"),
        CantRegister => ("H5E_CANTREGISTER", "Unable to register new ID"),
        CantRelease => ("H5E_CANTRELEASE", "Unable to release object"),
        CantRemove => ("H5E_CANTREMOVE", "Unable to remove object"),
        CantRename => ("H5E_CANTRENAME", "Unable to rename object"),
        CantReset => ("H5E_CANTRESET", "Can't reset object"),
        CantResize => ("H5E_CANTRESIZE", "Unable to resize a metadata cache entry"),
        CantRestore => ("H5E_CANTRESTORE", "Can't restore condition"),
        CantRevive => ("H5E_CANTREVIVE", "Can't revive object"),
        CantSelect => ("H5E_CANTSELECT", "Can't select hyperslab"),
        CantSerialize => ("H5E_CANTSERIALIZE", "Unable to serialize data from cache"),
        CantSet => ("H5E_CANTSET", "Can't set value"),
        CantShrink => ("H5E_CANTSHRINK", "Can't shrink container"),
        CantSort => ("H5E_CANTSORT", "Can't sort objects"),
        CantSplit => ("H5E_CANTSPLIT", "Unable to split node"),
        CantSwap => ("H5E_CANTSWAP", "Unable to swap records"),
        CantTag => ("H5E_CANTTAG", "Unable to tag metadata in the cache"),
        CantUncork => ("H5E_CANTUNCORK", "Unable to uncork an object"),
        CantUndepend => ("H5E_CANTUNDEPEND", "Unable to destroy a flush dependency"),
        CantUnlock => ("H5E_CANTUNLOCK", "Unable to unlock object"),
        CantUnlockFile => ("H5E_CANTUNLOCKFILE", "Unable to unlock file"),
        CantUnpin => ("H5E_CANTUNPIN", "Unable to un-pin cache entry"),
        CantUnprotect => ("H5E_CANTUNPROTECT", "Unable to unprotect metadata"),
        CantUnserialize => ("H5E_CANTUNSERIALIZE", "Unable to mark metadata as unserialized"),
        CantUpdate => ("H5E_CANTUPDATE", "Can't update object"),
        CantWait => ("H5E_CANTWAIT", "Can't wait on operation"),
        CloseError => ("H5E_CLOSEERROR", "Close failed"),
        CompLen => ("H5E_COMPLEN", "Name component is too long"),
        DupClass => ("H5E_DUPCLASS", "Duplicate class name in parent class"),
        Exists => ("H5E_EXISTS", "Object already exists"),
        Fcntl => ("H5E_FCNTL", "File control (fcntl) failed"),
        FileExists => ("H5E_FILEEXISTS", "File already exists"),
        FileOpen => ("H5E_FILEOPEN", "File already open"),
        InconsistentState => ("H5E_INCONSISTENTSTATE", "Internal states are inconsistent"),
        LinkCount => ("H5E_LINKCOUNT", "Bad object header link count"),
        [all(feature = "1.10.0", not(feature = "1.10.5"))] Logging => ("H5E_LOGFAIL", "Failure in the cache logging framework"),
        [not(all(feature = "1.10.0", not(feature = "1.10.5")))] Logging => ("H5E_LOGGING", "Failure in the cache logging framework"),
        Mount => ("H5E_MOUNT", "File mount error"),
        Mpi => ("H5E_MPI", "Some MPI function failed"),
        MpiErrStr => ("H5E_MPIERRSTR", "MPI Error String"),
        NLinks => ("H5E_NLINKS", "Too many soft links in path"),
        NoEncoder => ("H5E_NOENCODER", "Filter present but encoding disabled"),
        NoFilter => ("H5E_NOFILTER", "Requested filter is not available"),
        NoIds => ("H5E_NOIDS", "Out of IDs for group"),
        NoIndependent => ("H5E_NO_INDEPENDENT", "Can't perform independent IO"),
        NoSpace => ("H5E_NOSPACE", "No space available for allocation"),
        NoneMinor => ("H5E_NONE_MINOR", "No error"),
        NotCached => ("H5E_NOTCACHED", "Metadata not currently cached"),
        NotFound => ("H5E_NOTFOUND", "Object not found"),
        NotHdf5 => ("H5E_NOTHDF5", "Not an HDF5 file"),
        NotRegistered => ("H5E_NOTREGISTERED", "Link class not registered"),
        ObjOpen => ("H5E_OBJOPEN", "Object is already open"),
        OpenError => ("H5E_OPENERROR", "Can't open directory or file"),
        Overflow => ("H5E_OVERFLOW", "Address overflowed"),
        Path => ("H5E_PATH", "Problem with path to object"),
        Protect => ("H5E_PROTECT", "Protected metadata error"),
        ReadError => ("H5E_READERROR", "Read failed"),
        SeekError => ("H5E_SEEKERROR", "Seek failed"),
        SetDisallowed => ("H5E_SETDISALLOWED", "Disallowed operation"),
        SetLocal => ("H5E_SETLOCAL", "Error from filter 'set local' callback"),
        SysErrStr => ("H5E_SYSERRSTR", "System error message"),
        System => ("H5E_SYSTEM", "Internal error detected"),
        Traverse => ("H5E_TRAVERSE", "Link traversal failure"),
        Truncated => ("H5E_TRUNCATED", "File has been truncated"),
        Uninitialized => ("H5E_UNINITIALIZED", "Information is uinitialized"),
        Unmount => ("H5E_UNMOUNT", "File unmount error"),
        Unsupported => ("H5E_UNSUPPORTED", "Feature is unsupported"),
        Version => ("H5E_VERSION", "Wrong version number"),
        WriteError => ("H5E_WRITEERROR", "Write failed"),
    }
}
