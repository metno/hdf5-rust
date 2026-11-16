use hdf5_sys::h5t::{
    H5T_class_t, H5T_norm_t, H5T_pad_t, H5Tcopy, H5Tget_class, H5Tget_ebias, H5Tget_fields,
    H5Tget_inpad, H5Tget_norm, H5Tget_offset, H5Tget_order, H5Tget_pad, H5Tget_precision,
    H5Tget_size, H5Tset_ebias, H5Tset_fields, H5Tset_inpad, H5Tset_norm, H5Tset_offset,
    H5Tset_order, H5Tset_pad, H5Tset_precision, H5Tset_size,
};

use super::{ByteOrder, Datatype, Padding};
use crate::globals::{H5T_NATIVE_FLOAT, H5T_VAX_F32, H5T_VAX_F64};
use crate::internal_prelude::*;

/// The mantissa normalization of a floating-point datatype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Normalization {
    /// The most significant mantissa bit is always set and not stored.
    Implied,
    /// The most significant mantissa bit is always set and stored.
    MsbSet,
    /// The mantissa is not normalized.
    None,
}

impl From<Normalization> for H5T_norm_t {
    fn from(norm: Normalization) -> Self {
        match norm {
            Normalization::Implied => Self::H5T_NORM_IMPLIED,
            Normalization::MsbSet => Self::H5T_NORM_MSBSET,
            Normalization::None => Self::H5T_NORM_NONE,
        }
    }
}

impl TryFrom<H5T_norm_t> for Normalization {
    type Error = Error;

    fn try_from(norm: H5T_norm_t) -> Result<Self> {
        match norm {
            H5T_norm_t::H5T_NORM_IMPLIED => Ok(Self::Implied),
            H5T_norm_t::H5T_NORM_MSBSET => Ok(Self::MsbSet),
            H5T_norm_t::H5T_NORM_NONE => Ok(Self::None),
            H5T_norm_t::H5T_NORM_ERROR => fail!("Invalid floating-point normalization"),
        }
    }
}

/// The storage representation of a floating-point datatype.
///
/// Storage size is measured in bytes. Precision, offset, and field positions and
/// sizes are measured in bits. Field positions count from the least significant
/// bit of the storage, not from the start of the significant-bit range.
///
/// The significant-bit range fits within the storage size, and the sign, exponent,
/// and mantissa fields fit within the significant-bit range without overlapping.
/// Field widths match the limits of the HDF5 file format.
///
/// Construct a layout with [`builder`](Self::builder), or inspect an existing
/// datatype with [`Datatype::float_layout`].
///
/// # Examples
///
/// ```
/// use hdf5_metno::{Datatype, datatype::{ByteOrder, FloatLayout, Normalization}};
///
/// let half = FloatLayout::builder()
///     .size(2)
///     .byte_order(ByteOrder::BigEndian)
///     .bit_precision(16)
///     .sign_position(15)
///     .exponent_position(10)
///     .exponent_size(5)
///     .mantissa_position(0)
///     .mantissa_size(10)
///     .exponent_bias(15)
///     .normalization(Normalization::Implied)
///     .build()?;
///
/// let datatype = Datatype::from_float_layout(&half)?;
/// assert_eq!(datatype.float_layout()?, half);
/// # Ok::<(), hdf5_metno::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FloatLayout {
    size: u32,
    byte_order: ByteOrder,
    bit_offset: u16,
    bit_precision: u16,
    sign_position: u8,
    exponent_position: u8,
    exponent_size: u8,
    mantissa_position: u8,
    mantissa_size: u8,
    exponent_bias: u32,
    normalization: Normalization,
    low_padding: Padding,
    high_padding: Padding,
    internal_padding: Padding,
}

impl FloatLayout {
    /// Returns a builder for a floating-point layout.
    pub const fn builder() -> FloatLayoutBuilder {
        FloatLayoutBuilder::new()
    }

    /// Returns a builder initialized with all properties of this layout.
    pub const fn to_builder(&self) -> FloatLayoutBuilder {
        FloatLayoutBuilder {
            size: Some(self.size),
            byte_order: Some(self.byte_order),
            bit_offset: self.bit_offset,
            bit_precision: Some(self.bit_precision),
            sign_position: Some(self.sign_position),
            exponent_position: Some(self.exponent_position),
            exponent_size: Some(self.exponent_size),
            mantissa_position: Some(self.mantissa_position),
            mantissa_size: Some(self.mantissa_size),
            exponent_bias: Some(self.exponent_bias),
            normalization: Some(self.normalization),
            low_padding: self.low_padding,
            high_padding: self.high_padding,
            internal_padding: self.internal_padding,
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

    /// Returns the bit position of the sign bit.
    pub const fn sign_position(&self) -> u8 {
        self.sign_position
    }

    /// Returns the bit position of the least significant exponent bit.
    pub const fn exponent_position(&self) -> u8 {
        self.exponent_position
    }

    /// Returns the number of exponent bits.
    pub const fn exponent_size(&self) -> u8 {
        self.exponent_size
    }

    /// Returns the bit position of the least significant mantissa bit.
    pub const fn mantissa_position(&self) -> u8 {
        self.mantissa_position
    }

    /// Returns the number of mantissa bits.
    pub const fn mantissa_size(&self) -> u8 {
        self.mantissa_size
    }

    /// Returns the exponent bias.
    pub const fn exponent_bias(&self) -> u32 {
        self.exponent_bias
    }

    /// Returns the mantissa normalization.
    pub const fn normalization(&self) -> Normalization {
        self.normalization
    }

    /// Returns the padding of bits below the significant-bit range.
    pub const fn low_padding(&self) -> Padding {
        self.low_padding
    }

    /// Returns the padding of bits above the significant-bit range.
    pub const fn high_padding(&self) -> Padding {
        self.high_padding
    }

    /// Returns the padding of unused bits within the significant-bit range.
    pub const fn internal_padding(&self) -> Padding {
        self.internal_padding
    }
}

/// A builder for a [`FloatLayout`].
///
/// Size in bytes, byte order, precision, field positions and sizes, exponent bias,
/// and normalization must be supplied. The bit offset defaults to zero, and all
/// padding modes default to [`Padding::Zero`]. Start from
/// [`FloatLayout::to_builder`] to change only some properties of an existing layout.
///
/// [`build`](Self::build) checks the storage bounds and field placement. HDF5
/// checks support for the requested representation when the layout is applied to a
/// datatype.
#[derive(Clone, Debug)]
pub struct FloatLayoutBuilder {
    size: Option<u32>,
    byte_order: Option<ByteOrder>,
    bit_offset: u16,
    bit_precision: Option<u16>,
    sign_position: Option<u8>,
    exponent_position: Option<u8>,
    exponent_size: Option<u8>,
    mantissa_position: Option<u8>,
    mantissa_size: Option<u8>,
    exponent_bias: Option<u32>,
    normalization: Option<Normalization>,
    low_padding: Padding,
    high_padding: Padding,
    internal_padding: Padding,
}

impl Default for FloatLayoutBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl FloatLayoutBuilder {
    /// Creates a builder with zero offset and zero padding.
    ///
    /// All other properties are initially unset.
    pub const fn new() -> Self {
        Self {
            size: None,
            byte_order: None,
            bit_offset: 0,
            bit_precision: None,
            sign_position: None,
            exponent_position: None,
            exponent_size: None,
            mantissa_position: None,
            mantissa_size: None,
            exponent_bias: None,
            normalization: None,
            low_padding: Padding::Zero,
            high_padding: Padding::Zero,
            internal_padding: Padding::Zero,
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

    /// Sets the bit position of the sign bit.
    pub const fn sign_position(&mut self, position: u8) -> &mut Self {
        self.sign_position = Some(position);
        self
    }

    /// Sets the bit position of the least significant exponent bit.
    pub const fn exponent_position(&mut self, position: u8) -> &mut Self {
        self.exponent_position = Some(position);
        self
    }

    /// Sets the number of exponent bits.
    pub const fn exponent_size(&mut self, size: u8) -> &mut Self {
        self.exponent_size = Some(size);
        self
    }

    /// Sets the bit position of the least significant mantissa bit.
    pub const fn mantissa_position(&mut self, position: u8) -> &mut Self {
        self.mantissa_position = Some(position);
        self
    }

    /// Sets the number of mantissa bits.
    pub const fn mantissa_size(&mut self, size: u8) -> &mut Self {
        self.mantissa_size = Some(size);
        self
    }

    /// Sets the exponent bias.
    pub const fn exponent_bias(&mut self, bias: u32) -> &mut Self {
        self.exponent_bias = Some(bias);
        self
    }

    /// Sets the mantissa normalization.
    pub const fn normalization(&mut self, normalization: Normalization) -> &mut Self {
        self.normalization = Some(normalization);
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

    /// Sets the padding of unused bits within the significant-bit range.
    pub const fn internal_padding(&mut self, padding: Padding) -> &mut Self {
        self.internal_padding = padding;
        self
    }

    /// Builds a floating-point layout without modifying the builder.
    ///
    /// # Errors
    ///
    /// Returns an error if a required property is missing, size, precision, or a
    /// field size is zero, the significant-bit range exceeds the storage size, a
    /// field lies outside the significant-bit range, or fields overlap.
    pub fn build(&self) -> Result<FloatLayout> {
        let size = self.size.ok_or("Float storage size is required")?;
        let byte_order = self.byte_order.ok_or("Float byte order is required")?;
        let bit_precision = self.bit_precision.ok_or("Float bit precision is required")?;
        let sign_position = self.sign_position.ok_or("Float sign position is required")?;
        let exponent_position =
            self.exponent_position.ok_or("Float exponent position is required")?;
        let exponent_size = self.exponent_size.ok_or("Float exponent size is required")?;
        let mantissa_position =
            self.mantissa_position.ok_or("Float mantissa position is required")?;
        let mantissa_size = self.mantissa_size.ok_or("Float mantissa size is required")?;
        let exponent_bias = self.exponent_bias.ok_or("Float exponent bias is required")?;
        let normalization = self.normalization.ok_or("Float normalization is required")?;
        ensure!(size > 0, "Float storage size must be positive");
        ensure!(bit_precision > 0, "Float bit precision must be positive");
        ensure!(exponent_size > 0, "Float exponent size must be positive");
        ensure!(mantissa_size > 0, "Float mantissa size must be positive");
        let start = u32::from(self.bit_offset);
        let end = start + u32::from(bit_precision);
        ensure!(
            u64::from(end) <= u64::from(size) * 8,
            "Float significant bits exceed the storage size"
        );

        let significant = start..end;
        let sign = u32::from(sign_position);
        let exponent =
            u32::from(exponent_position)..u32::from(exponent_position) + u32::from(exponent_size);
        let mantissa =
            u32::from(mantissa_position)..u32::from(mantissa_position) + u32::from(mantissa_size);
        ensure!(significant.contains(&sign), "Float sign bit is outside the significant bits");
        ensure!(
            start <= exponent.start && exponent.end <= end,
            "Float exponent field is outside the significant bits"
        );
        ensure!(
            start <= mantissa.start && mantissa.end <= end,
            "Float mantissa field is outside the significant bits"
        );
        ensure!(!exponent.contains(&sign), "Float sign bit overlaps the exponent field");
        ensure!(!mantissa.contains(&sign), "Float sign bit overlaps the mantissa field");
        ensure!(
            exponent.end <= mantissa.start || mantissa.end <= exponent.start,
            "Float exponent and mantissa fields overlap"
        );

        Ok(FloatLayout {
            size,
            byte_order,
            bit_offset: self.bit_offset,
            bit_precision,
            sign_position,
            exponent_position,
            exponent_size,
            mantissa_position,
            mantissa_size,
            exponent_bias,
            normalization,
            low_padding: self.low_padding,
            high_padding: self.high_padding,
            internal_padding: self.internal_padding,
        })
    }
}

impl Datatype {
    /// Returns the storage layout of this floating-point datatype.
    ///
    /// Preserves properties that cannot be represented by
    /// [`to_descriptor`](Self::to_descriptor), including field positions, exponent
    /// bias, normalization, and padding.
    ///
    /// # Errors
    ///
    /// Returns an error if this is not a floating-point datatype, a property cannot
    /// be read or exceeds the HDF5 file format limits, or the reported layout fails
    /// the layout checks.
    pub fn float_layout(&self) -> Result<FloatLayout> {
        h5lock!({
            let id = self.id();
            ensure!(
                h5try!(H5Tget_class(id) as i32) == H5T_class_t::H5T_FLOAT as i32,
                "Expected a floating-point datatype"
            );
            let size = u32::try_from(h5try!(H5Tget_size(id) as hsize_t))
                .map_err(|_| "Float storage size exceeds the file format limit")?;
            let bit_precision = u16::try_from(h5try!(H5Tget_precision(id) as hsize_t))
                .map_err(|_| "Float bit precision exceeds the file format limit")?;
            let order = H5Tget_order(id);
            h5try!(order as i32);
            let bit_offset = u16::try_from(h5try!(H5Tget_offset(id)))
                .map_err(|_| "Float bit offset exceeds the file format limit")?;
            let (mut spos, mut epos, mut esize, mut mpos, mut msize) = (0, 0, 0, 0, 0);
            h5try!(H5Tget_fields(id, &mut spos, &mut epos, &mut esize, &mut mpos, &mut msize));
            let exponent_bias = u32::try_from(H5Tget_ebias(id))
                .map_err(|_| "Float exponent bias exceeds the file format limit")?;
            let normalization = H5Tget_norm(id);
            h5try!(normalization as i32);
            let mut low_padding = H5T_pad_t::H5T_PAD_ZERO;
            let mut high_padding = H5T_pad_t::H5T_PAD_ZERO;
            h5try!(H5Tget_pad(id, &mut low_padding, &mut high_padding));
            let internal_padding = H5Tget_inpad(id);
            h5try!(internal_padding as i32);
            FloatLayout::builder()
                .size(size)
                .byte_order(order.into())
                .bit_offset(bit_offset)
                .bit_precision(bit_precision)
                .sign_position(
                    u8::try_from(spos)
                        .map_err(|_| "Float sign position exceeds the file format limit")?,
                )
                .exponent_position(
                    u8::try_from(epos)
                        .map_err(|_| "Float exponent position exceeds the file format limit")?,
                )
                .exponent_size(
                    u8::try_from(esize)
                        .map_err(|_| "Float exponent size exceeds the file format limit")?,
                )
                .mantissa_position(
                    u8::try_from(mpos)
                        .map_err(|_| "Float mantissa position exceeds the file format limit")?,
                )
                .mantissa_size(
                    u8::try_from(msize)
                        .map_err(|_| "Float mantissa size exceeds the file format limit")?,
                )
                .exponent_bias(exponent_bias)
                .normalization(normalization.try_into()?)
                .low_padding(low_padding.try_into()?)
                .high_padding(high_padding.try_into()?)
                .internal_padding(internal_padding.try_into()?)
                .build()
        })
    }

    /// Creates a floating-point datatype with the specified layout.
    ///
    /// # Errors
    ///
    /// Returns an error if HDF5 cannot create the datatype or does not support
    /// the requested representation.
    pub fn from_float_layout(layout: &FloatLayout) -> Result<Self> {
        h5lock!(Self::copy_with_float_layout(*H5T_NATIVE_FLOAT, layout))
    }

    /// Returns a copy of this floating-point datatype with the specified layout.
    ///
    /// # Errors
    ///
    /// Returns an error if this is not a floating-point datatype, it cannot be
    /// copied, or HDF5 does not support the requested representation.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_metno::Datatype;
    ///
    /// let original = Datatype::from_type::<f32>()?;
    /// let layout = original.float_layout()?.to_builder().exponent_bias(126).build()?;
    ///
    /// let customized = original.with_float_layout(&layout)?;
    /// assert_eq!(customized.float_layout()?, layout);
    /// assert_eq!(original.float_layout()?.exponent_bias(), 127);
    /// # Ok::<(), hdf5_metno::Error>(())
    /// ```
    pub fn with_float_layout(&self, layout: &FloatLayout) -> Result<Self> {
        Self::copy_with_float_layout(self.id(), layout)
    }

    /// Returns a copy of the predefined 32-bit VAX floating-point datatype.
    ///
    /// # Errors
    ///
    /// Returns an error if HDF5 cannot copy the datatype.
    pub fn vax_f32() -> Result<Self> {
        h5lock!(Self::from_id(h5try!(H5Tcopy(*H5T_VAX_F32))))
    }

    /// Returns a copy of the predefined 64-bit VAX floating-point datatype.
    ///
    /// # Errors
    ///
    /// Returns an error if HDF5 cannot copy the datatype.
    pub fn vax_f64() -> Result<Self> {
        h5lock!(Self::from_id(h5try!(H5Tcopy(*H5T_VAX_F64))))
    }

    fn copy_with_float_layout(id: hid_t, layout: &FloatLayout) -> Result<Self> {
        let order = layout.byte_order.try_into()?;
        h5lock!({
            ensure!(
                h5try!(H5Tget_class(id) as i32) == H5T_class_t::H5T_FLOAT as i32,
                "Expected a floating-point datatype"
            );
            let datatype = Self::from_id(h5try!(H5Tcopy(id)))?;
            let id = datatype.id();
            // HDF5 rejects fields outside the current precision, so widen first and narrow last.
            let max_size = (layout.size as usize).max(h5try!(H5Tget_size(id) as hsize_t) as usize);
            h5try!(H5Tset_size(id, max_size));
            h5try!(H5Tset_offset(id, 0));
            h5try!(H5Tset_precision(id, max_size * 8));
            h5try!(H5Tset_fields(
                id,
                layout.sign_position.into(),
                layout.exponent_position.into(),
                layout.exponent_size.into(),
                layout.mantissa_position.into(),
                layout.mantissa_size.into()
            ));
            h5try!(H5Tset_offset(id, layout.bit_offset.into()));
            h5try!(H5Tset_precision(id, layout.bit_precision.into()));
            h5try!(H5Tset_size(id, layout.size as usize));
            h5try!(H5Tset_order(id, order));
            h5try!(H5Tset_ebias(id, layout.exponent_bias as usize));
            h5try!(H5Tset_norm(id, layout.normalization.into()));
            h5try!(H5Tset_pad(id, layout.low_padding.into(), layout.high_padding.into()));
            h5try!(H5Tset_inpad(id, layout.internal_padding.into()));
            Ok(datatype)
        })
    }
}
