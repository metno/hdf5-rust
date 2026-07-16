use std::convert::Infallible;
use std::error::Error as StdError;
use std::fmt;
use std::io;
use std::ops::Deref;
use std::panic;
use std::ptr::{self, addr_of_mut};

use ndarray::ShapeError;

use crate::error_codes::{MajorErrorCode, MinorErrorCode};

#[cfg(not(feature = "1.10.0"))]
use hdf5_sys::h5::hssize_t;
use hdf5_sys::h5e::{
    H5E_DEFAULT, H5E_WALK_DOWNWARD, H5E_auto2_t, H5E_error2_t, H5Eget_current_stack, H5Eget_msg,
    H5Eprint2, H5Eset_auto2, H5Ewalk2,
};

use crate::internal_prelude::*;

/// Silence errors emitted by `hdf5`
///
/// Safety: This version is not thread-safe and must be synchronised
/// with other calls to `hdf5`
pub(crate) unsafe fn silence_errors_no_sync(silence: bool) {
    // Cast function with different argument types. This is safe because H5Eprint2 is
    // documented to support this interface
    let h5eprint: Option<unsafe extern "C" fn(hid_t, *mut libc::FILE) -> herr_t> =
        Some(H5Eprint2 as _);
    let h5eprint: H5E_auto2_t = std::mem::transmute(h5eprint);
    H5Eset_auto2(H5E_DEFAULT, if silence { None } else { h5eprint }, ptr::null_mut());
}

/// Silence errors emitted by `hdf5`
pub fn silence_errors(silence: bool) {
    h5lock!(silence_errors_no_sync(silence));
}

/// A stack of error records from an HDF5 library call.
#[repr(transparent)]
#[derive(Clone)]
pub struct ErrorStack(Handle);

impl ObjectClass for ErrorStack {
    const NAME: &'static str = "errorstack";
    const VALID_TYPES: &'static [H5I_type_t] = &[H5I_ERROR_STACK];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }

    // TODO: short_repr()
}

impl ErrorStack {
    pub(crate) fn from_current() -> Result<Self> {
        let stack_id = h5lock!(H5Eget_current_stack());
        Handle::try_new(stack_id).map(Self)
    }

    /// Expands the error stack to a format which is easier to handle
    // known HDF5 bug: H5Eget_msg() used in this function may corrupt
    // the current stack, so we use self over &self
    pub fn expand(self) -> Result<ExpandedErrorStack> {
        struct CallbackData {
            stack: ExpandedErrorStack,
            err: Option<Error>,
        }
        unsafe extern "C" fn callback(
            _: c_uint, err_desc: *const H5E_error2_t, data: *mut c_void,
        ) -> herr_t {
            panic::catch_unwind(|| unsafe {
                let data = &mut *(data.cast::<CallbackData>());
                if data.err.is_some() {
                    return 0;
                }
                let closure = |e: H5E_error2_t| -> Result<ErrorFrame> {
                    let (desc, func) = (string_from_cstr(e.desc), string_from_cstr(e.func_name));
                    let major = get_h5_str(|m, s| H5Eget_msg(e.maj_num, ptr::null_mut(), m, s))?;
                    let minor = get_h5_str(|m, s| H5Eget_msg(e.min_num, ptr::null_mut(), m, s))?;
                    let major_code = MajorErrorCode::from_id(e.maj_num);
                    let minor_code = MinorErrorCode::from_id(e.min_num);
                    Ok(ErrorFrame::new(&desc, &func, &major, &minor, major_code, minor_code))
                };
                match closure(*err_desc) {
                    Ok(frame) => {
                        data.stack.push(frame);
                    }
                    Err(err) => {
                        data.err = Some(err);
                    }
                }
                0
            })
            .unwrap_or(-1)
        }

        let mut data = CallbackData { stack: ExpandedErrorStack::new(), err: None };
        let data_ptr: *mut c_void = addr_of_mut!(data).cast::<c_void>();

        let stack_id = self.handle().id();
        h5lock!({
            H5Ewalk2(stack_id, H5E_WALK_DOWNWARD, Some(callback), data_ptr);
        });

        data.err.map_or(Ok(data.stack), Err)
    }
}

/// An error record for an HDF5 library call.
#[derive(Clone, Debug)]
pub struct ErrorFrame {
    desc: String,
    func: String,
    major: String,
    minor: String,
    description: String,
    major_code: MajorErrorCode,
    minor_code: MinorErrorCode,
}

impl ErrorFrame {
    pub(crate) fn new(
        desc: &str, func: &str, major: &str, minor: &str, major_code: MajorErrorCode,
        minor_code: MinorErrorCode,
    ) -> Self {
        Self {
            desc: desc.into(),
            func: func.into(),
            major: major.into(),
            minor: minor.into(),
            description: format!("{func}(): {desc}"),
            major_code,
            minor_code,
        }
    }

    /// Returns the error description.
    pub fn desc(&self) -> &str {
        self.desc.as_ref()
    }

    /// Returns the major code: which part of the library failed.
    pub fn major_code(&self) -> MajorErrorCode {
        self.major_code
    }

    /// Returns the minor code: how the operation failed.
    pub fn minor_code(&self) -> MinorErrorCode {
        self.minor_code
    }

    /// Returns a message with the error description and the relevant function name.
    pub fn description(&self) -> &str {
        self.description.as_ref()
    }

    /// Returns a message with the error description and the relevant function name, file name,
    /// and line number.
    pub fn detail(&self) -> Option<String> {
        Some(format!("Error in {}(): {} [{}: {}]", self.func, self.desc, self.major, self.minor))
    }
}

/// A converted [`ErrorStack`] with methods to access [`ErrorFrame`] data.
#[derive(Clone, Debug)]
pub struct ExpandedErrorStack {
    frames: Vec<ErrorFrame>,
    description: Option<String>,
}

impl Deref for ExpandedErrorStack {
    type Target = [ErrorFrame];

    fn deref(&self) -> &Self::Target {
        &self.frames
    }
}

impl Default for ExpandedErrorStack {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpandedErrorStack {
    pub(crate) fn new() -> Self {
        Self { frames: Vec::new(), description: None }
    }

    pub(crate) fn push(&mut self, frame: ErrorFrame) {
        self.frames.push(frame);
        if !self.is_empty() {
            let top_desc = self.frames[0].description().to_owned();
            if self.len() == 1 {
                self.description = Some(top_desc);
            } else {
                self.description =
                    Some(format!("{}: {}", top_desc, self.frames[self.len() - 1].desc()));
            }
        }
    }

    /// Returns the top [`ErrorFrame`] of the stack, or `None` if it is empty.
    pub fn top(&self) -> Option<&ErrorFrame> {
        self.first()
    }

    /// Returns the major code of every frame, outermost first.
    pub fn major_codes(&self) -> impl Iterator<Item = MajorErrorCode> + '_ {
        self.frames.iter().map(ErrorFrame::major_code)
    }

    /// Returns the minor code of every frame, outermost first.
    pub fn minor_codes(&self) -> impl Iterator<Item = MinorErrorCode> + '_ {
        self.frames.iter().map(ErrorFrame::minor_code)
    }

    /// Returns whether any frame in the stack carries the given major code.
    pub fn contains_major(&self, code: MajorErrorCode) -> bool {
        self.major_codes().any(|c| c == code)
    }

    /// Returns whether any frame in the stack carries the given minor code.
    ///
    /// The cause is usually on an inner frame, so prefer this over [`top`](Self::top): opening
    /// a corrupt file reports `CantOpenFile` on the outermost frame and `NotHdf5` further down.
    pub fn contains_minor(&self, code: MinorErrorCode) -> bool {
        self.minor_codes().any(|c| c == code)
    }

    /// Returns the description of the error on top of the stack.
    pub fn description(&self) -> &str {
        match self.description {
            None => "unknown library error",
            Some(ref desc) => desc.as_ref(),
        }
    }

    /// Returns a detailed message for the error on top of the stack, or `None` if it is empty.
    pub fn detail(&self) -> Option<String> {
        self.top().and_then(ErrorFrame::detail)
    }
}

/// The error type for HDF5-related functions.
#[derive(Clone)]
pub enum Error {
    /// An error occurred in the C API of the HDF5 library. Full error stack is captured.
    HDF5(ErrorStack),
    /// A user error occurred in the high-level Rust API (e.g., invalid user input).
    Internal(String),
}

/// A type for results generated by HDF5-related functions where the `Err` type is
/// set to `hdf5::Error`.
pub type Result<T, E = Error> = ::std::result::Result<T, E>;

impl Error {
    /// Obtain the current error stack. The stack might be empty, which
    /// will result in a valid error stack
    pub fn query() -> Result<Self> {
        if let Ok(stack) = ErrorStack::from_current() {
            Ok(Self::HDF5(stack))
        } else {
            Err(Self::Internal("Could not get errorstack".to_owned()))
        }
    }

    /// Returns the expanded error stack, or `None` if there is none to expand.
    pub fn stack(&self) -> Option<ExpandedErrorStack> {
        match *self {
            Self::Internal(_) => None,
            Self::HDF5(ref stack) => stack.clone().expand().ok(),
        }
    }

    /// Returns whether any frame carries the given major code, `false` for
    /// [`Internal`](Self::Internal) errors.
    pub fn contains_major(&self, code: MajorErrorCode) -> bool {
        self.stack().is_some_and(|s| s.contains_major(code))
    }

    /// Returns whether any frame carries the given minor code, `false` for
    /// [`Internal`](Self::Internal) errors.
    ///
    /// ```
    /// use hdf5_metno as hdf5;
    /// use hdf5::MinorErrorCode;
    ///
    /// # let dir = tempfile::tempdir().unwrap();
    /// # let path = dir.path().join("corrupt.h5");
    /// # std::fs::write(&path, b"not an HDF5 file").unwrap();
    /// let err = hdf5::File::open(&path).unwrap_err();
    /// assert!(err.contains_minor(MinorErrorCode::NotHdf5));
    /// ```
    pub fn contains_minor(&self, code: MinorErrorCode) -> bool {
        self.stack().is_some_and(|s| s.contains_minor(code))
    }
}

impl From<&str> for Error {
    fn from(desc: &str) -> Self {
        Self::Internal(desc.into())
    }
}

impl From<String> for Error {
    fn from(desc: String) -> Self {
        Self::Internal(desc)
    }
}

impl From<Infallible> for Error {
    fn from(_: Infallible) -> Self {
        unreachable!("Infallible error can never be constructed")
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Self::Internal(ref desc) => f.write_str(desc),
            Self::HDF5(ref stack) => match stack.clone().expand() {
                Ok(stack) => f.write_str(stack.description()),
                Err(_) => f.write_str("Could not get error stack"),
            },
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Self::Internal(ref desc) => f.write_str(desc),
            Self::HDF5(ref stack) => match stack.clone().expand() {
                Ok(stack) => f.write_str(stack.description()),
                Err(_) => f.write_str("Could not get error stack"),
            },
        }
    }
}

impl StdError for Error {}

impl From<ShapeError> for Error {
    fn from(err: ShapeError) -> Self {
        format!("shape error: {err}").into()
    }
}

impl From<Error> for io::Error {
    fn from(err: Error) -> Self {
        Self::new(io::ErrorKind::Other, err)
    }
}

pub fn h5check<T: H5ErrorCode>(value: T) -> Result<T> {
    H5ErrorCode::h5check(value)
}

#[allow(unused)]
pub fn is_err_code<T: H5ErrorCode>(value: T) -> bool {
    H5ErrorCode::is_err_code(value)
}

pub trait H5ErrorCode: Copy {
    fn is_err_code(value: Self) -> bool;

    fn h5check(value: Self) -> Result<Self> {
        if Self::is_err_code(value) { Err(Error::query().unwrap_or_else(|e| e)) } else { Ok(value) }
    }
}

impl H5ErrorCode for hsize_t {
    fn is_err_code(value: Self) -> bool {
        value == 0
    }
}

impl H5ErrorCode for herr_t {
    fn is_err_code(value: Self) -> bool {
        value < 0
    }
}

#[cfg(feature = "1.10.0")]
impl H5ErrorCode for hid_t {
    fn is_err_code(value: Self) -> bool {
        value < 0
    }
}

#[cfg(not(feature = "1.10.0"))]
impl H5ErrorCode for hssize_t {
    fn is_err_code(value: Self) -> bool {
        value < 0
    }
}

impl H5ErrorCode for libc::ssize_t {
    fn is_err_code(value: Self) -> bool {
        value < 0
    }
}

#[cfg(test)]
pub mod tests {
    use hdf5_sys::h5p::{H5Pclose, H5Pcreate};

    use crate::error_codes::{MAJOR_CODES, MINOR_CODES};
    use crate::globals::{H5E_CANTOPENFILE, H5E_FILE, H5P_ROOT};
    use crate::internal_prelude::*;
    use crate::test::with_tmp_path;
    use crate::{MajorErrorCode, MinorErrorCode};
    use hdf5_sys::h5e::H5Eget_msg;
    use std::fs;
    use std::ptr;

    use super::ExpandedErrorStack;

    /// Checks every code this crate declares against the linked HDF5 itself: the description
    /// must be byte-identical to what `H5Eget_msg` returns for that id. This is what validates
    /// the transcribed table, and it fails if a variant is wired to the wrong global, if a
    /// description literal is wrong, or if a renamed code's `meta` gates overlap so that the
    /// wrong arm is selected.
    #[test]
    pub fn test_descriptions_match_the_linked_library() {
        fn msg_of(id: hid_t) -> String {
            h5lock!(get_h5_str(|m, s| H5Eget_msg(id, ptr::null_mut(), m, s))).unwrap()
        }

        assert!(!MAJOR_CODES.is_empty() && !MINOR_CODES.is_empty());
        for (&id, &code) in MAJOR_CODES.iter() {
            assert_eq!(code.description().unwrap(), msg_of(id), "{code:?}");
        }
        for (&id, &code) in MINOR_CODES.iter() {
            // H5E_LOGFAIL is upstream's binary-compatibility alias for H5E_LOGGING and carries
            // its own message. Both ids resolve to Logging, so only the canonical one can
            // match, therefor we pin the alias's message (instead of skipping it).
            #[cfg(all(feature = "1.10.5", not(feature = "1.12.0")))]
            if id == *crate::globals::H5E_LOGFAIL {
                assert_eq!(code, MinorErrorCode::Logging);
                assert_eq!(msg_of(id), "old H5E_LOGGING_g (maintained for binary compatibility)");
                continue;
            }
            assert_eq!(code.description().unwrap(), msg_of(id), "{code:?}");
        }
    }

    /// Checks the codes captured for a real frame vs. that frame's own text.
    /// The test above checks the id -> description mapping, this one validates that `expand`
    /// converts the right id to the right enum.
    #[test]
    pub fn test_frame_codes_agree_with_frame_text() {
        with_tmp_path(|path| {
            fs::write(&path, b"garbage data").unwrap();
            let err = File::open(&path).unwrap_err();
            let stack = err.stack().unwrap();
            assert!(stack.len() >= 3, "expected a multi-frame stack, got {}", stack.len());

            for frame in stack.iter() {
                // detail() renders "Error in f(): desc [<major msg>: <minor msg>]" from the strings H5Eget_msg returned.
                let major_code_desc = frame.major_code().description().unwrap();
                let minor_code_desc = frame.minor_code().description().unwrap();
                let maj_min_desc = format!("[{major_code_desc}: {minor_code_desc}]");
                let detail = frame.detail().unwrap();
                assert!(
                    detail.ends_with(&maj_min_desc),
                    "{detail:?} does not end with {maj_min_desc:?}"
                );
            }
            assert_eq!(stack.major_codes().count(), stack.len());
            assert_eq!(stack.minor_codes().count(), stack.len());
        });
    }

    /// Every id the linked HDF5 registers must map to exactly one variant, and no two variants
    /// may share an id. The reverse is not asserted: a variant whose symbol this HDF5 lacks is
    /// deliberately unreachable.
    #[test]
    pub fn test_symbol_mapping_is_unambiguous() {
        for (&id, &code) in MAJOR_CODES.iter() {
            assert!(MajorErrorCode::all().contains(&code), "{code:?} is not a declared variant");
            assert_ne!(id, H5I_INVALID_HID);
        }
        for (&id, &code) in MINOR_CODES.iter() {
            assert!(MinorErrorCode::all().contains(&code), "{code:?} is not a declared variant");
            assert_ne!(id, H5I_INVALID_HID);
        }
        // the map is populated: these two exist in every supported HDF5
        assert_eq!(MajorErrorCode::from_id(*H5E_FILE), MajorErrorCode::File);
        assert_eq!(MinorErrorCode::from_id(*H5E_CANTOPENFILE), MinorErrorCode::CantOpenFile);
    }

    #[test]
    pub fn test_unknown_error_code_is_preserved() {
        // H5I_INVALID_HID is never a registered message id
        let minor = MinorErrorCode::from_id(H5I_INVALID_HID);
        assert_eq!(minor, MinorErrorCode::Other(H5I_INVALID_HID));
        assert_eq!(minor.name(), None);
        assert_eq!(minor.description(), None);
        assert_eq!(minor.to_string(), format!("unknown error code ({H5I_INVALID_HID})"));

        let major = MajorErrorCode::from_id(H5I_INVALID_HID);
        assert_eq!(major, MajorErrorCode::Other(H5I_INVALID_HID));
        assert_eq!(major.name(), None);
        assert_eq!(major.description(), None);
        assert_eq!(major.to_string(), format!("unknown error code ({H5I_INVALID_HID})"));

        // a major id must not resolve as a minor code, or vice versa
        assert_eq!(MinorErrorCode::from_id(*H5E_FILE), MinorErrorCode::Other(*H5E_FILE));
        assert_eq!(
            MajorErrorCode::from_id(*H5E_CANTOPENFILE),
            MajorErrorCode::Other(*H5E_CANTOPENFILE)
        );
    }

    #[test]
    pub fn test_error_codes_on_stack() {
        let err = h5lock!({
            let plist_id = H5Pcreate(*H5P_ROOT);
            H5Pclose(plist_id);
            H5Pclose(plist_id);
            Error::query()
        })
        .unwrap();
        let stack = err.stack().unwrap();
        let top = stack.top().unwrap();
        assert_eq!(top.major_code(), MajorErrorCode::PropertyList);
        assert_eq!(top.major_code().name(), Some("H5E_PLIST"));
        assert_eq!(top.minor_code(), MinorErrorCode::CantFree);
        assert!(err.contains_major(MajorErrorCode::PropertyList));
        assert!(!err.contains_major(MajorErrorCode::File));
        // codes are resolved for every frame, not just the top one
        assert!(stack.major_codes().all(|c| !matches!(c, MajorErrorCode::Other(_))));
        assert!(stack.minor_codes().all(|c| !matches!(c, MinorErrorCode::Other(_))));
    }

    /// The three file-open failure modes must be distinguishable by code, not by message text.
    #[test]
    pub fn test_file_error_codes_are_distinguishable() {
        // a corrupt file is reported as NotHdf5 by an inner frame
        with_tmp_path(|path| {
            fs::write(&path, b"garbage data").unwrap();
            let err = File::open_rw(&path).unwrap_err();
            assert!(err.contains_minor(MinorErrorCode::NotHdf5), "{err:?}");
            assert!(err.contains_major(MajorErrorCode::File));
            // the VOL frames in a file-open stack must resolve to a named code (not Other)
            #[cfg(feature = "1.12.0")]
            assert!(err.contains_major(MajorErrorCode::VirtualObjectLayer), "{err:?}");
            let stack = err.stack().unwrap();
            assert!(stack.major_codes().all(|c| !matches!(c, MajorErrorCode::Other(_))), "{err:?}");
            assert!(stack.minor_codes().all(|c| !matches!(c, MinorErrorCode::Other(_))), "{err:?}");
        });

        // a missing file reports CantOpenFile
        with_tmp_path(|path| {
            let err = File::open_rw(&path).unwrap_err();
            assert!(err.contains_minor(MinorErrorCode::CantOpenFile), "{err:?}");
            assert!(!err.contains_minor(MinorErrorCode::NotHdf5), "{err:?}");
        });

        // Creating exclusively over an existing file reports CantCreate/CantOpenFile, not H5E_FILEEXISTS.
        // the EEXIST is only visible as errno text in the innermost frame.
        // Codes therefore cannot tell "already exists" apart from any other open failure.
        with_tmp_path(|path| {
            File::create(&path).unwrap();
            let err = File::create_excl(&path).unwrap_err();
            assert!(err.contains_minor(MinorErrorCode::CantCreate), "{err:?}");
            assert!(!err.contains_minor(MinorErrorCode::NotHdf5), "{err:?}");
        });
    }

    /// Internal errors carry no HDF5 stack.
    #[test]
    pub fn test_internal_error_has_no_codes() {
        let err = Error::Internal("nope".into());
        assert!(err.stack().is_none());
        assert!(!err.contains_minor(MinorErrorCode::NotHdf5));
        assert!(!err.contains_major(MajorErrorCode::File));
    }

    #[test]
    pub fn test_error_code_display() {
        assert_eq!(MinorErrorCode::NotHdf5.to_string(), "Not an HDF5 file");
        assert_eq!(MajorErrorCode::File.to_string(), "File accessibility");
    }

    #[test]
    pub fn test_error_stack() {
        let stack = h5lock!({
            let plist_id = H5Pcreate(*H5P_ROOT);
            H5Pclose(plist_id);
            Error::query()
        })
        .unwrap();
        let stack = match stack {
            Error::HDF5(stack) => stack,
            Error::Internal(internal) => panic!("Expected hdf5 error, not {}", internal),
        }
        .expand()
        .unwrap();
        assert!(stack.is_empty());

        let stack = h5lock!({
            let plist_id = H5Pcreate(*H5P_ROOT);
            H5Pclose(plist_id);
            H5Pclose(plist_id);
            Error::query()
        })
        .unwrap();
        let stack = match stack {
            Error::HDF5(stack) => stack,
            Error::Internal(internal) => panic!("Expected hdf5 error, not {}", internal),
        }
        .expand()
        .unwrap();
        assert_eq!(stack.description(), "H5Pclose(): can't close: can't locate ID");
        assert_eq!(
            &stack.detail().unwrap(),
            "Error in H5Pclose(): can't close [Property lists: Unable to free object]"
        );

        assert!(stack.len() >= 2 && stack.len() <= 4); // depending on HDF5 version
        assert!(!stack.is_empty());

        assert_eq!(stack[0].description(), "H5Pclose(): can't close");
        assert_eq!(
            &stack[0].detail().unwrap(),
            "Error in H5Pclose(): can't close \
             [Property lists: Unable to free object]"
        );

        #[cfg(not(feature = "1.14.0"))]
        {
            assert_eq!(stack[stack.len() - 1].description(), "H5I_dec_ref(): can't locate ID");
            assert_eq!(
                &stack[stack.len() - 1].detail().unwrap(),
                "Error in H5I_dec_ref(): can't locate ID \
             [Object atom: Unable to find atom information (already closed?)]"
            );
        }
        #[cfg(feature = "1.14.0")]
        {
            assert_eq!(stack[stack.len() - 1].description(), "H5I__dec_ref(): can't locate ID");
            assert_eq!(
                &stack[stack.len() - 1].detail().unwrap(),
                "Error in H5I__dec_ref(): can't locate ID \
             [Object ID: Unable to find ID information (already closed?)]"
            );
        }

        let empty_stack = ExpandedErrorStack::new();
        assert!(empty_stack.is_empty());
        assert_eq!(empty_stack.len(), 0);
    }

    #[test]
    pub fn test_h5call() {
        let result_no_error = h5call!({
            let plist_id = H5Pcreate(*H5P_ROOT);
            H5Pclose(plist_id)
        });
        assert!(result_no_error.is_ok());

        let result_error = h5call!({
            let plist_id = H5Pcreate(*H5P_ROOT);
            H5Pclose(plist_id);
            H5Pclose(plist_id)
        });
        assert!(result_error.is_err());
    }

    #[test]
    pub fn test_h5try() {
        fn f1() -> Result<herr_t> {
            h5try!(H5Pcreate(*H5P_ROOT));
            Ok(100)
        }

        assert_eq!(f1().unwrap(), 100);

        fn f2() -> Result<herr_t> {
            h5try!(H5Pcreate(123456));
            Ok(100)
        }

        assert!(f2().is_err());
    }
}
