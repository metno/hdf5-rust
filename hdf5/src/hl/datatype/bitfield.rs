use hdf5_sys::h5t::{
    H5T_class_t, H5T_pad_t, H5Tcopy, H5Tget_class, H5Tget_offset, H5Tget_order, H5Tget_pad,
    H5Tget_precision, H5Tget_size, H5Tset_offset, H5Tset_order, H5Tset_pad, H5Tset_precision,
    H5Tset_size,
};

use super::{ByteOrder, Datatype, Padding};
use crate::globals::H5T_NATIVE_B8;
use crate::internal_prelude::*;

/// The storage representation of a bitfield datatype.
///
/// Storage size is measured in bytes. Precision and offset are measured in bits.
/// The significant-bit range fits within the storage size.
///
/// Construct a layout with [`builder`](Self::builder), or inspect an existing
/// datatype with [`Datatype::bitfield_layout`].
///
/// # Examples
///
/// ```
/// use hdf5_metno::{Datatype, datatype::{BitfieldLayout, ByteOrder, Padding}};
///
/// let layout = BitfieldLayout::builder()
///     .size(3)
///     .byte_order(ByteOrder::BigEndian)
///     .bit_offset(2)
///     .bit_precision(20)
///     .low_padding(Padding::Zero)
///     .high_padding(Padding::One)
///     .build()?;
///
/// let datatype = Datatype::from_bitfield_layout(&layout)?;
/// assert_eq!(datatype.bitfield_layout()?, layout);
/// # Ok::<(), hdf5_metno::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitfieldLayout {
    size: u32,
    byte_order: ByteOrder,
    bit_offset: u16,
    bit_precision: u16,
    low_padding: Padding,
    high_padding: Padding,
}

impl BitfieldLayout {
    /// Returns a builder for a bitfield layout.
    pub const fn builder() -> BitfieldLayoutBuilder {
        BitfieldLayoutBuilder::new()
    }

    /// Returns a builder initialized with all properties of this layout.
    pub const fn to_builder(&self) -> BitfieldLayoutBuilder {
        BitfieldLayoutBuilder {
            size: Some(self.size),
            byte_order: Some(self.byte_order),
            bit_offset: self.bit_offset,
            bit_precision: Some(self.bit_precision),
            low_padding: self.low_padding,
            high_padding: self.high_padding,
        }
    }

    /// Returns the storage size in bytes.
    pub const fn size(&self) -> u32 {
        self.size
    }

    /// Returns the byte order.
    pub const fn byte_order(&self) -> ByteOrder {
        self.byte_order
    }

    /// Returns the number of bits preceding the significant-bit range.
    pub const fn bit_offset(&self) -> u16 {
        self.bit_offset
    }

    /// Returns the number of significant bits.
    pub const fn bit_precision(&self) -> u16 {
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

/// A builder for a [`BitfieldLayout`].
///
/// Size in bytes, byte order, and precision must be supplied. The bit offset defaults
/// to zero, and both padding modes default to [`Padding::Zero`].
///
/// [`build`](Self::build) checks the storage bounds. HDF5 checks support for the
/// requested representation when the layout is applied to a datatype.
#[derive(Clone, Debug)]
pub struct BitfieldLayoutBuilder {
    size: Option<u32>,
    byte_order: Option<ByteOrder>,
    bit_offset: u16,
    bit_precision: Option<u16>,
    low_padding: Padding,
    high_padding: Padding,
}

impl Default for BitfieldLayoutBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BitfieldLayoutBuilder {
    /// Creates a builder with zero offset and zero padding.
    ///
    /// Size, byte order, and precision are initially unset.
    pub const fn new() -> Self {
        Self {
            size: None,
            byte_order: None,
            bit_offset: 0,
            bit_precision: None,
            low_padding: Padding::Zero,
            high_padding: Padding::Zero,
        }
    }

    /// Sets the storage size in bytes.
    pub const fn size(&mut self, size: u32) -> &mut Self {
        self.size = Some(size);
        self
    }

    /// Sets the byte order.
    pub const fn byte_order(&mut self, byte_order: ByteOrder) -> &mut Self {
        self.byte_order = Some(byte_order);
        self
    }

    /// Sets the number of bits preceding the significant-bit range.
    pub const fn bit_offset(&mut self, bit_offset: u16) -> &mut Self {
        self.bit_offset = bit_offset;
        self
    }

    /// Sets the number of significant bits.
    pub const fn bit_precision(&mut self, bit_precision: u16) -> &mut Self {
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

    /// Builds a bitfield layout without modifying the builder.
    ///
    /// # Errors
    ///
    /// Returns an error if a required property is missing, size or precision is
    /// zero, or the significant-bit range exceeds the storage size.
    pub fn build(&self) -> Result<BitfieldLayout> {
        let size = self.size.ok_or("Bitfield storage size is required")?;
        let byte_order = self.byte_order.ok_or("Bitfield byte order is required")?;
        let bit_precision = self.bit_precision.ok_or("Bitfield bit precision is required")?;
        ensure!(size > 0, "Bitfield storage size must be positive");
        ensure!(bit_precision > 0, "Bitfield bit precision must be positive");
        let end = u64::from(self.bit_offset) + u64::from(bit_precision);
        ensure!(end <= u64::from(size) * 8, "Bitfield significant bits exceed the storage size");
        Ok(BitfieldLayout {
            size,
            byte_order,
            bit_offset: self.bit_offset,
            bit_precision,
            low_padding: self.low_padding,
            high_padding: self.high_padding,
        })
    }
}

impl Datatype {
    /// Returns the storage layout of this bitfield datatype.
    ///
    /// # Errors
    ///
    /// Returns an error if this is not a bitfield datatype, a property cannot be
    /// read or exceeds the HDF5 file format limits, or the reported layout fails
    /// the storage-bound checks.
    pub fn bitfield_layout(&self) -> Result<BitfieldLayout> {
        h5lock!({
            ensure!(
                h5try!(H5Tget_class(self.id()) as i32) == H5T_class_t::H5T_BITFIELD as i32,
                "Expected a bitfield datatype"
            );
            let size = u32::try_from(h5try!(H5Tget_size(self.id()) as hsize_t))
                .map_err(|_| "Bitfield storage size exceeds the file format limit")?;
            let bit_precision = u16::try_from(h5try!(H5Tget_precision(self.id()) as hsize_t))
                .map_err(|_| "Bitfield bit precision exceeds the file format limit")?;
            let order = H5Tget_order(self.id());
            h5try!(order as i32);
            let bit_offset = u16::try_from(h5try!(H5Tget_offset(self.id())))
                .map_err(|_| "Bitfield bit offset exceeds the file format limit")?;
            let mut low_padding = H5T_pad_t::H5T_PAD_ZERO;
            let mut high_padding = H5T_pad_t::H5T_PAD_ZERO;
            h5try!(H5Tget_pad(self.id(), &mut low_padding, &mut high_padding));
            BitfieldLayout::builder()
                .size(size)
                .byte_order(order.into())
                .bit_offset(bit_offset)
                .bit_precision(bit_precision)
                .low_padding(low_padding.try_into()?)
                .high_padding(high_padding.try_into()?)
                .build()
        })
    }

    /// Creates a bitfield datatype with the specified layout.
    ///
    /// # Errors
    ///
    /// Returns an error if HDF5 cannot create the datatype or does not support
    /// the requested representation.
    pub fn from_bitfield_layout(layout: &BitfieldLayout) -> Result<Self> {
        h5lock!(Self::copy_with_bitfield_layout(*H5T_NATIVE_B8, layout))
    }

    /// Returns a copy of this bitfield datatype with the specified layout.
    ///
    /// # Errors
    ///
    /// Returns an error if this is not a bitfield datatype, it cannot be copied,
    /// or HDF5 does not support the requested representation.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_metno::{Datatype, datatype::{BitfieldLayout, ByteOrder}};
    ///
    /// let layout = BitfieldLayout::builder()
    ///     .size(2)
    ///     .byte_order(ByteOrder::LittleEndian)
    ///     .bit_precision(16)
    ///     .build()?;
    /// let original = Datatype::from_bitfield_layout(&layout)?;
    ///
    /// let big_endian = layout.to_builder().byte_order(ByteOrder::BigEndian).build()?;
    /// let customized = original.with_bitfield_layout(&big_endian)?;
    /// assert_eq!(customized.bitfield_layout()?, big_endian);
    /// assert_eq!(original.bitfield_layout()?, layout);
    /// # Ok::<(), hdf5_metno::Error>(())
    /// ```
    pub fn with_bitfield_layout(&self, layout: &BitfieldLayout) -> Result<Self> {
        Self::copy_with_bitfield_layout(self.id(), layout)
    }

    fn copy_with_bitfield_layout(id: hid_t, layout: &BitfieldLayout) -> Result<Self> {
        let order = layout.byte_order.try_into()?;
        h5lock!({
            ensure!(
                h5try!(H5Tget_class(id) as i32) == H5T_class_t::H5T_BITFIELD as i32,
                "Expected a bitfield datatype"
            );
            let datatype = Self::from_id(h5try!(H5Tcopy(id)))?;
            h5try!(H5Tset_offset(datatype.id(), 0));
            h5try!(H5Tset_precision(datatype.id(), 1));
            h5try!(H5Tset_size(datatype.id(), layout.size as usize));
            h5try!(H5Tset_precision(datatype.id(), layout.bit_precision.into()));
            h5try!(H5Tset_offset(datatype.id(), layout.bit_offset.into()));
            h5try!(H5Tset_order(datatype.id(), order));
            h5try!(H5Tset_pad(
                datatype.id(),
                layout.low_padding.into(),
                layout.high_padding.into()
            ));
            Ok(datatype)
        })
    }
}
