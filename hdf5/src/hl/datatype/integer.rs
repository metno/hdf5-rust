use hdf5_sys::h5t::{
    H5T_class_t, H5T_pad_t, H5T_sign_t, H5Tcopy, H5Tget_class, H5Tget_offset, H5Tget_order,
    H5Tget_pad, H5Tget_precision, H5Tget_sign, H5Tget_size, H5Tset_offset, H5Tset_order,
    H5Tset_pad, H5Tset_precision, H5Tset_sign, H5Tset_size,
};

use super::{ByteOrder, Datatype, Padding};
use crate::globals::H5T_NATIVE_INT8;
use crate::internal_prelude::*;

/// The storage representation of an integer datatype.
///
/// Storage size is measured in bytes. Precision and offset are measured in bits.
/// The significant-bit range fits within the storage size.
///
/// Construct a layout with [`builder`](Self::builder), or inspect an existing
/// datatype with [`Datatype::integer_layout`].
///
/// # Examples
///
/// ```
/// use hdf5_metno::{Datatype, datatype::{ByteOrder, IntegerLayout, Padding}};
///
/// let layout = IntegerLayout::builder()
///     .size(3)
///     .byte_order(ByteOrder::BigEndian)
///     .signed(false)
///     .bit_offset(2)
///     .bit_precision(20)
///     .low_padding(Padding::Zero)
///     .high_padding(Padding::One)
///     .build()?;
///
/// let datatype = Datatype::from_integer_layout(&layout)?;
/// assert_eq!(datatype.integer_layout()?, layout);
/// # Ok::<(), hdf5_metno::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntegerLayout {
    size: usize,
    byte_order: ByteOrder,
    signed: bool,
    bit_offset: usize,
    bit_precision: usize,
    low_padding: Padding,
    high_padding: Padding,
}

impl IntegerLayout {
    /// Returns a builder for an integer layout.
    pub const fn builder() -> IntegerLayoutBuilder {
        IntegerLayoutBuilder::new()
    }

    /// Returns a builder initialized with all properties of this layout.
    pub const fn to_builder(&self) -> IntegerLayoutBuilder {
        IntegerLayoutBuilder {
            size: Some(self.size),
            byte_order: Some(self.byte_order),
            signed: Some(self.signed),
            bit_offset: self.bit_offset,
            bit_precision: Some(self.bit_precision),
            low_padding: self.low_padding,
            high_padding: self.high_padding,
        }
    }

    /// Returns the storage size in bytes.
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Returns the byte order.
    pub const fn byte_order(&self) -> ByteOrder {
        self.byte_order
    }

    /// Returns whether the integer is signed.
    pub const fn signed(&self) -> bool {
        self.signed
    }

    /// Returns the number of bits preceding the significant-bit range.
    pub const fn bit_offset(&self) -> usize {
        self.bit_offset
    }

    /// Returns the number of significant bits.
    pub const fn bit_precision(&self) -> usize {
        self.bit_precision
    }

    /// Returns the padding of bits below the significant-bit range.
    pub const fn low_padding(&self) -> Padding {
        self.low_padding
    }

    /// Returns the padding of bits above the significant-bit range.
    pub const fn high_padding(&self) -> Padding {
        self.high_padding
    }
}

/// A builder for an [`IntegerLayout`].
///
/// Size in bytes, byte order, signedness, and precision must be supplied. The bit offset
/// defaults to zero, and both padding modes default to [`Padding::Zero`].
///
/// [`build`](Self::build) checks the storage bounds. HDF5 checks support for the
/// requested representation when the layout is applied to a datatype.
#[derive(Clone, Debug)]
pub struct IntegerLayoutBuilder {
    size: Option<usize>,
    byte_order: Option<ByteOrder>,
    signed: Option<bool>,
    bit_offset: usize,
    bit_precision: Option<usize>,
    low_padding: Padding,
    high_padding: Padding,
}

impl Default for IntegerLayoutBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl IntegerLayoutBuilder {
    /// Creates a builder with zero offset and zero padding.
    ///
    /// Size, byte order, signedness, and precision are initially unset.
    pub const fn new() -> Self {
        Self {
            size: None,
            byte_order: None,
            signed: None,
            bit_offset: 0,
            bit_precision: None,
            low_padding: Padding::Zero,
            high_padding: Padding::Zero,
        }
    }

    /// Sets the storage size in bytes.
    pub const fn size(&mut self, size: usize) -> &mut Self {
        self.size = Some(size);
        self
    }

    /// Sets the byte order.
    pub const fn byte_order(&mut self, byte_order: ByteOrder) -> &mut Self {
        self.byte_order = Some(byte_order);
        self
    }

    /// Sets whether the integer is signed.
    pub const fn signed(&mut self, signed: bool) -> &mut Self {
        self.signed = Some(signed);
        self
    }

    /// Sets the number of bits preceding the significant-bit range.
    pub const fn bit_offset(&mut self, bit_offset: usize) -> &mut Self {
        self.bit_offset = bit_offset;
        self
    }

    /// Sets the number of significant bits.
    pub const fn bit_precision(&mut self, bit_precision: usize) -> &mut Self {
        self.bit_precision = Some(bit_precision);
        self
    }

    /// Sets the padding of bits below the significant-bit range.
    pub const fn low_padding(&mut self, padding: Padding) -> &mut Self {
        self.low_padding = padding;
        self
    }

    /// Sets the padding of bits above the significant-bit range.
    pub const fn high_padding(&mut self, padding: Padding) -> &mut Self {
        self.high_padding = padding;
        self
    }

    /// Builds an integer layout without modifying the builder.
    ///
    /// # Errors
    ///
    /// Returns an error if a required property is missing, size or precision is
    /// zero, the significant-bit range exceeds the storage size, or computing
    /// the bit range or storage size in bits overflows.
    pub fn build(&self) -> Result<IntegerLayout> {
        let size = self.size.ok_or("Integer storage size is required")?;
        let byte_order = self.byte_order.ok_or("Integer byte order is required")?;
        let signed = self.signed.ok_or("Integer signedness is required")?;
        let bit_precision = self.bit_precision.ok_or("Integer bit precision is required")?;
        ensure!(size > 0, "Integer storage size must be positive");
        ensure!(bit_precision > 0, "Integer bit precision must be positive");
        let storage_bits = size.checked_mul(8).ok_or("Integer storage size overflows")?;
        let end = self
            .bit_offset
            .checked_add(bit_precision)
            .ok_or("Integer significant-bit range overflows")?;
        ensure!(end <= storage_bits, "Integer significant bits exceed the storage size");
        Ok(IntegerLayout {
            size,
            byte_order,
            signed,
            bit_offset: self.bit_offset,
            bit_precision,
            low_padding: self.low_padding,
            high_padding: self.high_padding,
        })
    }
}

impl Datatype {
    /// Returns the storage layout of this integer datatype.
    ///
    /// Preserves properties that cannot be represented by
    /// [`to_descriptor`](Self::to_descriptor), including nonstandard storage sizes,
    /// precision, offset, and padding.
    ///
    /// # Errors
    ///
    /// Returns an error if this is not an integer datatype, a property cannot be
    /// read, or the reported layout fails the storage-bound checks.
    pub fn integer_layout(&self) -> Result<IntegerLayout> {
        h5lock!({
            ensure!(
                h5try!(H5Tget_class(self.id()) as i32) == H5T_class_t::H5T_INTEGER as i32,
                "Expected an integer datatype"
            );
            let size = h5try!(H5Tget_size(self.id()) as hsize_t) as usize;
            let bit_precision = h5try!(H5Tget_precision(self.id()) as hsize_t) as usize;
            let order = H5Tget_order(self.id());
            h5try!(order as i32);
            let byte_order = order.into();
            let sign = H5Tget_sign(self.id());
            h5try!(sign as i32);
            let signed = match sign {
                H5T_sign_t::H5T_SGN_NONE => false,
                H5T_sign_t::H5T_SGN_2 => true,
                _ => fail!("Invalid integer signedness"),
            };
            let bit_offset = h5try!(H5Tget_offset(self.id())) as usize;
            let mut low_padding = H5T_pad_t::H5T_PAD_ZERO;
            let mut high_padding = H5T_pad_t::H5T_PAD_ZERO;
            h5try!(H5Tget_pad(self.id(), &mut low_padding, &mut high_padding));
            IntegerLayout::builder()
                .size(size)
                .byte_order(byte_order)
                .signed(signed)
                .bit_offset(bit_offset)
                .bit_precision(bit_precision)
                .low_padding(low_padding.try_into()?)
                .high_padding(high_padding.try_into()?)
                .build()
        })
    }

    /// Creates an integer datatype with the specified layout.
    ///
    /// # Errors
    ///
    /// Returns an error if HDF5 cannot create the datatype or does not support
    /// the requested representation.
    pub fn from_integer_layout(layout: &IntegerLayout) -> Result<Self> {
        h5lock!(Self::copy_with_integer_layout(*H5T_NATIVE_INT8, layout))
    }

    /// Returns a copy of this integer datatype with the specified layout.
    ///
    /// # Errors
    ///
    /// Returns an error if this is not an integer datatype, it cannot be copied,
    /// or HDF5 does not support the requested representation.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_metno::Datatype;
    ///
    /// let original = Datatype::from_type::<u32>()?;
    /// let layout = original.integer_layout()?
    ///     .to_builder()
    ///     .size(3)
    ///     .bit_precision(20)
    ///     .bit_offset(2)
    ///     .build()?;
    ///
    /// let customized = original.with_integer_layout(&layout)?;
    /// assert_eq!(customized.integer_layout()?, layout);
    /// assert_eq!(original.size(), 4);
    /// # Ok::<(), hdf5_metno::Error>(())
    /// ```
    pub fn with_integer_layout(&self, layout: &IntegerLayout) -> Result<Self> {
        Self::copy_with_integer_layout(self.id(), layout)
    }

    fn copy_with_integer_layout(id: hid_t, layout: &IntegerLayout) -> Result<Self> {
        let order = layout.byte_order.try_into()?;
        h5lock!({
            ensure!(
                h5try!(H5Tget_class(id) as i32) == H5T_class_t::H5T_INTEGER as i32,
                "Expected an integer datatype"
            );
            let datatype = Self::from_id(h5try!(H5Tcopy(id)))?;
            h5try!(H5Tset_offset(datatype.id(), 0));
            h5try!(H5Tset_precision(datatype.id(), 1));
            h5try!(H5Tset_size(datatype.id(), layout.size));
            h5try!(H5Tset_precision(datatype.id(), layout.bit_precision));
            h5try!(H5Tset_offset(datatype.id(), layout.bit_offset));
            h5try!(H5Tset_order(datatype.id(), order));
            let sign = if layout.signed { H5T_sign_t::H5T_SGN_2 } else { H5T_sign_t::H5T_SGN_NONE };
            h5try!(H5Tset_sign(datatype.id(), sign));
            h5try!(H5Tset_pad(
                datatype.id(),
                layout.low_padding.into(),
                layout.high_padding.into()
            ));
            Ok(datatype)
        })
    }
}
