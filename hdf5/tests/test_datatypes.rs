#[macro_use]
mod common;

use hdf5::datatype::{
    BitfieldLayout, ByteOrder, CompoundMember, FloatLayout, IntegerLayout, Normalization, Padding,
    StringPadding,
};
use hdf5::types::{TypeDescriptor as TD, *};
use hdf5::{Datatype, H5Type, MajorErrorCode, MinorErrorCode, from_id};
use hdf5_metno as hdf5;
use hdf5_sys::h5i::H5I_INVALID_HID;
use hdf5_sys::h5t::{H5T_class_t, H5Tget_class, H5Tset_fields, H5Tset_offset, H5Tset_precision};
use pretty_assertions::{assert_eq, assert_str_eq};

use self::common::util::new_in_memory_file;

macro_rules! check_roundtrip {
    ($ty:ty, $desc:expr) => {{
        let desc = <$ty as H5Type>::type_descriptor();
        assert_eq!(desc, $desc);
        let dt = Datatype::from_type::<$ty>().unwrap();
        assert_eq!(desc, dt.to_descriptor().unwrap());
        assert_eq!(dt.size(), desc.size());
    }};
}

#[test]
pub fn test_datatype_roundtrip() {
    check_roundtrip!(i8, TD::Integer(IntSize::U1));
    check_roundtrip!(i16, TD::Integer(IntSize::U2));
    check_roundtrip!(i32, TD::Integer(IntSize::U4));
    check_roundtrip!(i64, TD::Integer(IntSize::U8));
    check_roundtrip!(u8, TD::Unsigned(IntSize::U1));
    check_roundtrip!(u16, TD::Unsigned(IntSize::U2));
    check_roundtrip!(u32, TD::Unsigned(IntSize::U4));
    check_roundtrip!(u64, TD::Unsigned(IntSize::U8));
    #[cfg(feature = "f16")]
    check_roundtrip!(::half::f16, TD::Float(FloatSize::U2));
    check_roundtrip!(f32, TD::Float(FloatSize::U4));
    check_roundtrip!(f64, TD::Float(FloatSize::U8));
    check_roundtrip!(bool, TD::Boolean);
    check_roundtrip!([bool; 5], TD::FixedArray(Box::new(TD::Boolean), 5));
    check_roundtrip!(VarLenArray<bool>, TD::VarLenArray(Box::new(TD::Boolean)));
    check_roundtrip!(FixedAscii<5>, TD::FixedAscii(5));
    check_roundtrip!(FixedUnicode<5>, TD::FixedUnicode(5));
    check_roundtrip!(VarLenAscii, TD::VarLenAscii);
    check_roundtrip!(VarLenUnicode, TD::VarLenUnicode);

    #[allow(dead_code)]
    #[derive(H5Type)]
    #[repr(i64)]
    enum X {
        A = 1,
        B = -2,
    }
    let x_desc = TD::Enum(EnumType {
        size: IntSize::U8,
        signed: true,
        members: vec![
            EnumMember { name: "A".into(), value: 1 },
            EnumMember { name: "B".into(), value: -2i64 as _ },
        ],
    });
    check_roundtrip!(X, x_desc);

    #[allow(dead_code)]
    #[derive(H5Type)]
    #[repr(i64)]
    enum Y {
        #[hdf5(rename = "variant.a")]
        A = 1,
        B = -2,
    }
    let y_desc = TD::Enum(EnumType {
        size: IntSize::U8,
        signed: true,
        members: vec![
            EnumMember { name: "variant.a".into(), value: 1 },
            EnumMember { name: "B".into(), value: -2i64 as _ },
        ],
    });
    check_roundtrip!(Y, y_desc);

    #[derive(H5Type)]
    #[repr(C)]
    struct A {
        a: i64,
        b: u64,
    }
    let a_desc = TD::Compound(CompoundType {
        fields: vec![
            CompoundField::typed::<i64>("a", 0, 0),
            CompoundField::typed::<u64>("b", 8, 1),
        ],
        size: 16,
    });
    check_roundtrip!(A, a_desc);

    #[derive(H5Type)]
    #[repr(C)]
    struct C {
        a: [X; 2],
        b: [[A; 4]; 32],
    }
    let a_arr_desc = TD::FixedArray(Box::new(x_desc), 2);
    let b_arr_desc = TD::FixedArray(Box::new(TD::FixedArray(Box::new(a_desc), 4)), 32);
    let c_desc = TD::Compound(CompoundType {
        fields: vec![
            CompoundField::new("a", a_arr_desc, 0, 0),
            CompoundField::new("b", b_arr_desc, 16, 1),
        ],
        size: 2 * 8 + 4 * 32 * 16,
    });
    check_roundtrip!(C, c_desc);

    #[derive(H5Type)]
    #[repr(C)]
    struct D {
        #[hdf5(rename = "field.one")]
        a: f64,
        #[hdf5(rename = "field.two")]
        b: u64,
    }
    let d_desc = TD::Compound(CompoundType {
        fields: vec![
            CompoundField::typed::<f64>("field.one", 0, 0),
            CompoundField::typed::<u64>("field.two", 8, 1),
        ],
        size: 16,
    });
    check_roundtrip!(D, d_desc);

    #[derive(H5Type)]
    #[repr(C)]
    struct E(#[hdf5(rename = "alpha")] u64, f64);
    let e_desc = TD::Compound(CompoundType {
        fields: vec![
            CompoundField::typed::<u64>("alpha", 0, 0),
            CompoundField::typed::<f64>("1", 8, 1),
        ],
        size: 16,
    });
    check_roundtrip!(E, e_desc);
}

#[test]
pub fn test_with_byte_order_numeric_types() -> hdf5::Result<()> {
    let integer = Datatype::from_type::<i32>()?;
    let float = Datatype::from_type::<f64>()?;

    let little_integer = integer.with_byte_order(ByteOrder::LittleEndian)?;
    let big_integer = integer.with_byte_order(ByteOrder::BigEndian)?;
    let little_float = float.with_byte_order(ByteOrder::LittleEndian)?;
    let big_float = float.with_byte_order(ByteOrder::BigEndian)?;

    assert_eq!(little_integer.byte_order(), ByteOrder::LittleEndian);
    assert_eq!(big_integer.byte_order(), ByteOrder::BigEndian);
    assert_eq!(little_float.byte_order(), ByteOrder::LittleEndian);
    assert_eq!(big_float.byte_order(), ByteOrder::BigEndian);

    assert_eq!(little_integer.to_descriptor()?, integer.to_descriptor()?);
    assert_eq!(big_integer.to_descriptor()?, integer.to_descriptor()?);
    assert_eq!(little_float.to_descriptor()?, float.to_descriptor()?);
    assert_eq!(big_float.to_descriptor()?, float.to_descriptor()?);
    Ok(())
}

#[test]
pub fn test_with_byte_order_preserves_original_and_clone() -> hdf5::Result<()> {
    let original = Datatype::from_type::<u32>()?;
    let alias = original.clone();
    let original_order = original.byte_order();
    let other_order = if original_order == ByteOrder::LittleEndian {
        ByteOrder::BigEndian
    } else {
        ByteOrder::LittleEndian
    };

    let configured = alias.with_byte_order(other_order)?;
    assert_eq!(configured.byte_order(), other_order);
    assert_eq!(original.byte_order(), original_order);
    assert_eq!(alias.byte_order(), original_order);

    let error = alias.with_byte_order(ByteOrder::None).unwrap_err();
    assert!(error.contains_major(MajorErrorCode::Datatype), "{error:?}");
    assert!(error.contains_minor(MinorErrorCode::BadValue), "{error:?}");
    assert_eq!(original.byte_order(), original_order);
    assert_eq!(alias.byte_order(), original_order);
    Ok(())
}

#[test]
pub fn test_with_byte_order_committed_datatype() -> hdf5::Result<()> {
    let file = new_in_memory_file()?;
    let original = Datatype::from_type::<u32>()?.with_byte_order(ByteOrder::LittleEndian)?;
    file.commit_datatype("little_endian", &original)?;
    let committed = file.committed_datatype("little_endian")?;

    let configured = committed.as_datatype().with_byte_order(ByteOrder::BigEndian)?;

    assert!(!configured.is_committed());
    assert_eq!(configured.byte_order(), ByteOrder::BigEndian);
    assert!(committed.as_datatype().is_committed());
    assert_eq!(committed.as_datatype().byte_order(), ByteOrder::LittleEndian);
    Ok(())
}

#[test]
pub fn test_opaque_datatype_class() -> hdf5::Result<()> {
    let datatype = Datatype::opaque(16)?;

    let class = hdf5::h5lock!(H5Tget_class(datatype.id()));

    assert_eq!(class, H5T_class_t::H5T_OPAQUE);
    Ok(())
}

#[test]
pub fn test_opaque_datatype_rejects_zero_size() {
    let error = Datatype::opaque(0).unwrap_err();

    let hdf5::Error::Internal(message) = error else {
        panic!("Expected a size validation error, got {error:?}");
    };
    assert_str_eq!(message, "Opaque datatype size must be positive");
}

#[test]
pub fn test_integer_layout_rejects_bits_beyond_storage() {
    let error = IntegerLayout::builder()
        .size(3)
        .byte_order(ByteOrder::BigEndian)
        .signed(false)
        .bit_precision(20)
        .bit_offset(5)
        .build()
        .unwrap_err();
    let hdf5::Error::Internal(message) = error else {
        panic!("Expected a layout validation error, got {error:?}");
    };
    assert_str_eq!(message, "Integer significant bits exceed the storage size");
}

#[test]
pub fn test_with_integer_layout_does_not_modify_original() -> hdf5::Result<()> {
    let original = Datatype::from_type::<i32>()?;
    let alias = original.clone();
    let original_layout = original.integer_layout()?;
    let layout = original_layout.to_builder().signed(false).build()?;

    let configured = alias.with_integer_layout(&layout)?;

    assert!(!configured.integer_layout()?.signed());
    assert_eq!(original.integer_layout()?, original_layout);
    assert_eq!(alias.integer_layout()?, original_layout);
    Ok(())
}

#[test]
pub fn test_with_integer_layout_shrinks_storage() -> hdf5::Result<()> {
    let original = Datatype::from_type::<u64>()?;
    let layout = original.integer_layout()?.to_builder().size(3).bit_precision(24).build()?;

    let configured = original.with_integer_layout(&layout)?;

    assert_eq!(configured.integer_layout()?, layout);
    Ok(())
}

#[test]
pub fn test_bitfield_layout_rejects_bits_beyond_storage() {
    let error = BitfieldLayout::builder()
        .size(3)
        .byte_order(ByteOrder::BigEndian)
        .bit_precision(20)
        .bit_offset(5)
        .build()
        .unwrap_err();
    let hdf5::Error::Internal(message) = error else {
        panic!("Expected a layout validation error, got {error:?}");
    };
    assert_str_eq!(message, "Bitfield significant bits exceed the storage size");
}

#[test]
pub fn test_bitfield_layout_requires_bit_precision() {
    let error =
        BitfieldLayout::builder().size(2).byte_order(ByteOrder::LittleEndian).build().unwrap_err();
    let hdf5::Error::Internal(message) = error else {
        panic!("Expected a layout validation error, got {error:?}");
    };
    assert_str_eq!(message, "Bitfield bit precision is required");
}

#[test]
pub fn test_bitfield_layout_rejects_integer_datatype() -> hdf5::Result<()> {
    let integer = Datatype::from_type::<u16>()?;

    let error = integer.bitfield_layout().unwrap_err();

    let hdf5::Error::Internal(message) = error else {
        panic!("Expected a datatype class error, got {error:?}");
    };
    assert_str_eq!(message, "Expected a bitfield datatype");
    Ok(())
}

#[test]
pub fn test_with_bitfield_layout_rejects_integer_datatype() -> hdf5::Result<()> {
    let integer = Datatype::from_type::<u16>()?;
    let layout = BitfieldLayout::builder()
        .size(2)
        .byte_order(ByteOrder::LittleEndian)
        .bit_precision(16)
        .build()?;

    let error = integer.with_bitfield_layout(&layout).unwrap_err();

    let hdf5::Error::Internal(message) = error else {
        panic!("Expected a datatype class error, got {error:?}");
    };
    assert_str_eq!(message, "Expected a bitfield datatype");
    Ok(())
}

#[test]
pub fn test_with_bitfield_layout_does_not_modify_original() -> hdf5::Result<()> {
    let original_layout = BitfieldLayout::builder()
        .size(2)
        .byte_order(ByteOrder::LittleEndian)
        .bit_precision(16)
        .build()?;
    let original = Datatype::from_bitfield_layout(&original_layout)?;
    let alias = original.clone();
    let layout = original_layout
        .to_builder()
        .byte_order(ByteOrder::BigEndian)
        .high_padding(Padding::One)
        .build()?;

    let configured = alias.with_bitfield_layout(&layout)?;

    assert_eq!(configured.bitfield_layout()?, layout);
    assert_eq!(original.bitfield_layout()?, original_layout);
    assert_eq!(alias.bitfield_layout()?, original_layout);
    Ok(())
}

#[test]
pub fn test_with_bitfield_layout_shrinks_storage() -> hdf5::Result<()> {
    let wide = BitfieldLayout::builder()
        .size(8)
        .byte_order(ByteOrder::LittleEndian)
        .bit_precision(64)
        .build()?;
    let original = Datatype::from_bitfield_layout(&wide)?;
    let layout = wide.to_builder().size(3).bit_precision(24).build()?;

    let configured = original.with_bitfield_layout(&layout)?;

    assert_eq!(configured.bitfield_layout()?, layout);
    Ok(())
}

#[test]
pub fn test_with_bitfield_layout_rejects_none_byte_order() -> hdf5::Result<()> {
    let original_layout = BitfieldLayout::builder()
        .size(2)
        .byte_order(ByteOrder::LittleEndian)
        .bit_precision(16)
        .build()?;
    let original = Datatype::from_bitfield_layout(&original_layout)?;
    let layout = original_layout.to_builder().byte_order(ByteOrder::None).build()?;

    let error = original.with_bitfield_layout(&layout).unwrap_err();

    assert!(error.contains_major(MajorErrorCode::Datatype), "{error:?}");
    assert!(error.contains_minor(MinorErrorCode::BadValue), "{error:?}");
    assert_eq!(original.bitfield_layout()?, original_layout);
    Ok(())
}

#[test]
pub fn test_with_bitfield_layout_committed_datatype() -> hdf5::Result<()> {
    let file = new_in_memory_file()?;
    let little_endian = BitfieldLayout::builder()
        .size(2)
        .byte_order(ByteOrder::LittleEndian)
        .bit_precision(16)
        .build()?;
    file.commit_datatype("bitfield", &Datatype::from_bitfield_layout(&little_endian)?)?;
    let committed = file.committed_datatype("bitfield")?;
    let big_endian = little_endian.to_builder().byte_order(ByteOrder::BigEndian).build()?;

    let configured = committed.as_datatype().with_bitfield_layout(&big_endian)?;

    assert!(!configured.is_committed());
    assert_eq!(configured.bitfield_layout()?, big_endian);
    assert!(committed.as_datatype().is_committed());
    assert_eq!(committed.as_datatype().bitfield_layout()?, little_endian);
    Ok(())
}

#[test]
pub fn test_bitfield_layout_rejects_offset_beyond_file_format() -> hdf5::Result<()> {
    let layout = BitfieldLayout::builder()
        .size(1)
        .byte_order(ByteOrder::LittleEndian)
        .bit_precision(8)
        .build()?;
    let datatype = Datatype::from_bitfield_layout(&layout)?;
    hdf5::h5call!(H5Tset_offset(datatype.id(), 70_000))?;

    let error = datatype.bitfield_layout().unwrap_err();

    let hdf5::Error::Internal(message) = error else {
        panic!("Expected a file format limit error, got {error:?}");
    };
    assert_str_eq!(message, "Bitfield bit offset exceeds the file format limit");
    Ok(())
}

#[test]
pub fn test_bitfield_layout_rejects_precision_beyond_file_format() -> hdf5::Result<()> {
    let layout = BitfieldLayout::builder()
        .size(1)
        .byte_order(ByteOrder::LittleEndian)
        .bit_precision(8)
        .build()?;
    let datatype = Datatype::from_bitfield_layout(&layout)?;
    hdf5::h5call!(H5Tset_precision(datatype.id(), 70_000))?;

    let error = datatype.bitfield_layout().unwrap_err();

    let hdf5::Error::Internal(message) = error else {
        panic!("Expected a file format limit error, got {error:?}");
    };
    assert_str_eq!(message, "Bitfield bit precision exceeds the file format limit");
    Ok(())
}

#[test]
pub fn test_integer_layout_rejects_offset_beyond_file_format() -> hdf5::Result<()> {
    let datatype = Datatype::from_type::<u8>()?;
    hdf5::h5call!(H5Tset_offset(datatype.id(), 70_000))?;

    let error = datatype.integer_layout().unwrap_err();

    let hdf5::Error::Internal(message) = error else {
        panic!("Expected a file format limit error, got {error:?}");
    };
    assert_str_eq!(message, "Integer bit offset exceeds the file format limit");
    Ok(())
}

#[test]
pub fn test_integer_layout_rejects_precision_beyond_file_format() -> hdf5::Result<()> {
    let datatype = Datatype::from_type::<u8>()?;
    hdf5::h5call!(H5Tset_precision(datatype.id(), 70_000))?;

    let error = datatype.integer_layout().unwrap_err();

    let hdf5::Error::Internal(message) = error else {
        panic!("Expected a file format limit error, got {error:?}");
    };
    assert_str_eq!(message, "Integer bit precision exceeds the file format limit");
    Ok(())
}

#[test]
pub fn test_float_layout_rejects_bits_beyond_storage() -> hdf5::Result<()> {
    let f32_layout = Datatype::from_type::<f32>()?.float_layout()?;

    let error = f32_layout.to_builder().size(3).build().unwrap_err();

    let hdf5::Error::Internal(message) = error else {
        panic!("Expected a layout validation error, got {error:?}");
    };
    assert_str_eq!(message, "Float significant bits exceed the storage size");
    Ok(())
}

#[test]
pub fn test_float_layout_requires_normalization() {
    let error = FloatLayout::builder()
        .size(2)
        .byte_order(ByteOrder::LittleEndian)
        .bit_precision(16)
        .sign_position(15)
        .exponent_position(10)
        .exponent_size(5)
        .mantissa_position(0)
        .mantissa_size(10)
        .exponent_bias(15)
        .build()
        .unwrap_err();
    let hdf5::Error::Internal(message) = error else {
        panic!("Expected a layout validation error, got {error:?}");
    };
    assert_str_eq!(message, "Float normalization is required");
}

#[test]
pub fn test_float_layout_rejects_zero_exponent_size() -> hdf5::Result<()> {
    let f32_layout = Datatype::from_type::<f32>()?.float_layout()?;

    let error = f32_layout.to_builder().exponent_size(0).build().unwrap_err();

    let hdf5::Error::Internal(message) = error else {
        panic!("Expected a layout validation error, got {error:?}");
    };
    assert_str_eq!(message, "Float exponent size must be positive");
    Ok(())
}

#[test]
pub fn test_float_layout_rejects_zero_mantissa_size() -> hdf5::Result<()> {
    let f32_layout = Datatype::from_type::<f32>()?.float_layout()?;

    let error = f32_layout.to_builder().mantissa_size(0).build().unwrap_err();

    let hdf5::Error::Internal(message) = error else {
        panic!("Expected a layout validation error, got {error:?}");
    };
    assert_str_eq!(message, "Float mantissa size must be positive");
    Ok(())
}

#[test]
pub fn test_float_layout_rejects_sign_outside_significant_bits() -> hdf5::Result<()> {
    let f32_layout = Datatype::from_type::<f32>()?.float_layout()?;

    let error = f32_layout.to_builder().bit_precision(31).build().unwrap_err();

    let hdf5::Error::Internal(message) = error else {
        panic!("Expected a layout validation error, got {error:?}");
    };
    assert_str_eq!(message, "Float sign bit is outside the significant bits");
    Ok(())
}

#[test]
pub fn test_float_layout_rejects_exponent_outside_significant_bits() -> hdf5::Result<()> {
    let f32_layout = Datatype::from_type::<f32>()?.float_layout()?;

    let error = f32_layout.to_builder().exponent_position(25).build().unwrap_err();

    let hdf5::Error::Internal(message) = error else {
        panic!("Expected a layout validation error, got {error:?}");
    };
    assert_str_eq!(message, "Float exponent field is outside the significant bits");
    Ok(())
}

#[test]
pub fn test_float_layout_rejects_mantissa_outside_significant_bits() -> hdf5::Result<()> {
    let f32_layout = Datatype::from_type::<f32>()?.float_layout()?;

    let error = f32_layout.to_builder().mantissa_position(10).build().unwrap_err();

    let hdf5::Error::Internal(message) = error else {
        panic!("Expected a layout validation error, got {error:?}");
    };
    assert_str_eq!(message, "Float mantissa field is outside the significant bits");
    Ok(())
}

#[test]
pub fn test_float_layout_rejects_sign_in_exponent() -> hdf5::Result<()> {
    let f32_layout = Datatype::from_type::<f32>()?.float_layout()?;

    let error = f32_layout.to_builder().sign_position(23).build().unwrap_err();

    let hdf5::Error::Internal(message) = error else {
        panic!("Expected a layout validation error, got {error:?}");
    };
    assert_str_eq!(message, "Float sign bit overlaps the exponent field");
    Ok(())
}

#[test]
pub fn test_float_layout_rejects_sign_in_mantissa() -> hdf5::Result<()> {
    let f32_layout = Datatype::from_type::<f32>()?.float_layout()?;

    let error = f32_layout.to_builder().sign_position(0).build().unwrap_err();

    let hdf5::Error::Internal(message) = error else {
        panic!("Expected a layout validation error, got {error:?}");
    };
    assert_str_eq!(message, "Float sign bit overlaps the mantissa field");
    Ok(())
}

#[test]
pub fn test_float_layout_rejects_overlapping_exponent_and_mantissa() -> hdf5::Result<()> {
    let f32_layout = Datatype::from_type::<f32>()?.float_layout()?;

    let error = f32_layout.to_builder().mantissa_size(24).build().unwrap_err();

    let hdf5::Error::Internal(message) = error else {
        panic!("Expected a layout validation error, got {error:?}");
    };
    assert_str_eq!(message, "Float exponent and mantissa fields overlap");
    Ok(())
}

#[test]
pub fn test_float_layout_of_f32() -> hdf5::Result<()> {
    let byte_order =
        if cfg!(target_endian = "little") { ByteOrder::LittleEndian } else { ByteOrder::BigEndian };
    let expected = FloatLayout::builder()
        .size(4)
        .byte_order(byte_order)
        .bit_precision(32)
        .sign_position(31)
        .exponent_position(23)
        .exponent_size(8)
        .mantissa_position(0)
        .mantissa_size(23)
        .exponent_bias(127)
        .normalization(Normalization::Implied)
        .build()?;

    let layout = Datatype::from_type::<f32>()?.float_layout()?;

    assert_eq!(layout, expected);
    Ok(())
}

#[test]
pub fn test_float_layout_of_f64() -> hdf5::Result<()> {
    let byte_order =
        if cfg!(target_endian = "little") { ByteOrder::LittleEndian } else { ByteOrder::BigEndian };
    let expected = FloatLayout::builder()
        .size(8)
        .byte_order(byte_order)
        .bit_precision(64)
        .sign_position(63)
        .exponent_position(52)
        .exponent_size(11)
        .mantissa_position(0)
        .mantissa_size(52)
        .exponent_bias(1023)
        .normalization(Normalization::Implied)
        .build()?;

    let layout = Datatype::from_type::<f64>()?.float_layout()?;

    assert_eq!(layout, expected);
    Ok(())
}

#[test]
pub fn test_vax_f32_layout() -> hdf5::Result<()> {
    let byte_order = ByteOrder::Vax;
    let expected = FloatLayout::builder()
        .size(4)
        .byte_order(byte_order)
        .bit_precision(32)
        .sign_position(31)
        .exponent_position(23)
        .exponent_size(8)
        .mantissa_position(0)
        .mantissa_size(23)
        .exponent_bias(129)
        .normalization(Normalization::Implied)
        .build()?;

    let layout = Datatype::vax_f32()?.float_layout()?;

    assert_eq!(layout, expected);
    Ok(())
}

#[test]
pub fn test_vax_f64_layout() -> hdf5::Result<()> {
    let byte_order = ByteOrder::Vax;
    let expected = FloatLayout::builder()
        .size(8)
        .byte_order(byte_order)
        .bit_precision(64)
        .sign_position(63)
        .exponent_position(52)
        .exponent_size(11)
        .mantissa_position(0)
        .mantissa_size(52)
        .exponent_bias(1025)
        .normalization(Normalization::Implied)
        .build()?;

    let layout = Datatype::vax_f64()?.float_layout()?;

    assert_eq!(layout, expected);
    Ok(())
}

#[test]
pub fn test_float_layout_rejects_integer_datatype() -> hdf5::Result<()> {
    let integer = Datatype::from_type::<u32>()?;

    let error = integer.float_layout().unwrap_err();

    let hdf5::Error::Internal(message) = error else {
        panic!("Expected a datatype class error, got {error:?}");
    };
    assert_str_eq!(message, "Expected a floating-point datatype");
    Ok(())
}

#[test]
pub fn test_float_layout_rejects_exponent_position_beyond_file_format() -> hdf5::Result<()> {
    let datatype = Datatype::from_type::<f32>()?;
    hdf5::h5call!(H5Tset_precision(datatype.id(), 320))?;
    hdf5::h5call!(H5Tset_fields(datatype.id(), 31, 300, 8, 0, 23))?;

    let error = datatype.float_layout().unwrap_err();

    let hdf5::Error::Internal(message) = error else {
        panic!("Expected a file format limit error, got {error:?}");
    };
    assert_str_eq!(message, "Float exponent position exceeds the file format limit");
    Ok(())
}

#[test]
pub fn test_with_float_layout_rejects_integer_datatype() -> hdf5::Result<()> {
    let integer = Datatype::from_type::<u32>()?;
    let layout = Datatype::from_type::<f32>()?.float_layout()?;

    let error = integer.with_float_layout(&layout).unwrap_err();

    let hdf5::Error::Internal(message) = error else {
        panic!("Expected a datatype class error, got {error:?}");
    };
    assert_str_eq!(message, "Expected a floating-point datatype");
    Ok(())
}

#[test]
pub fn test_with_float_layout_does_not_modify_original() -> hdf5::Result<()> {
    let original = Datatype::from_type::<f32>()?;
    let alias = original.clone();
    let original_layout = original.float_layout()?;
    let layout = original_layout.to_builder().exponent_bias(126).build()?;

    let configured = alias.with_float_layout(&layout)?;

    assert_eq!(configured.float_layout()?, layout);
    assert_eq!(original.float_layout()?, original_layout);
    assert_eq!(alias.float_layout()?, original_layout);
    Ok(())
}

#[test]
pub fn test_with_float_layout_shrinks_storage() -> hdf5::Result<()> {
    let original = Datatype::from_type::<f64>()?;
    let half = FloatLayout::builder()
        .size(2)
        .byte_order(ByteOrder::BigEndian)
        .bit_precision(16)
        .sign_position(15)
        .exponent_position(10)
        .exponent_size(5)
        .mantissa_position(0)
        .mantissa_size(10)
        .exponent_bias(15)
        .normalization(Normalization::Implied)
        .build()?;

    let configured = original.with_float_layout(&half)?;

    assert_eq!(configured.float_layout()?, half);
    Ok(())
}

#[test]
pub fn test_with_float_layout_grows_storage_with_offset() -> hdf5::Result<()> {
    let original = Datatype::from_type::<f32>()?;
    let layout = original
        .float_layout()?
        .to_builder()
        .size(5)
        .bit_offset(4)
        .sign_position(35)
        .exponent_position(27)
        .mantissa_position(4)
        .low_padding(Padding::One)
        .internal_padding(Padding::One)
        .normalization(Normalization::MsbSet)
        .build()?;

    let configured = original.with_float_layout(&layout)?;

    assert_eq!(configured.float_layout()?, layout);
    Ok(())
}

#[test]
pub fn test_with_float_layout_rejects_none_byte_order() -> hdf5::Result<()> {
    let original = Datatype::from_type::<f32>()?;
    let original_layout = original.float_layout()?;
    let layout = original_layout.to_builder().byte_order(ByteOrder::None).build()?;

    let error = original.with_float_layout(&layout).unwrap_err();

    assert!(error.contains_major(MajorErrorCode::Datatype), "{error:?}");
    assert!(error.contains_minor(MinorErrorCode::BadValue), "{error:?}");
    assert_eq!(original.float_layout()?, original_layout);
    Ok(())
}

#[test]
pub fn test_with_float_layout_committed_datatype() -> hdf5::Result<()> {
    let file = new_in_memory_file()?;
    file.commit_datatype("float", &Datatype::from_type::<f32>()?)?;
    let committed = file.committed_datatype("float")?;
    let original_layout = committed.as_datatype().float_layout()?;
    let layout = original_layout.to_builder().exponent_bias(126).build()?;

    let configured = committed.as_datatype().with_float_layout(&layout)?;

    assert!(!configured.is_committed());
    assert_eq!(configured.float_layout()?, layout);
    assert!(committed.as_datatype().is_committed());
    assert_eq!(committed.as_datatype().float_layout()?, original_layout);
    Ok(())
}

#[test]
pub fn test_invalid_datatype() {
    assert_err!(from_id::<Datatype>(H5I_INVALID_HID), "Invalid handle id");
}

#[test]
pub fn test_committed_datatype_roundtrip() -> hdf5::Result<()> {
    let compound = TD::Compound(CompoundType {
        fields: vec![
            CompoundField::typed::<i32>("a", 0, 0),
            CompoundField::typed::<f64>("b", 8, 1),
        ],
        size: 16,
    });
    let enumeration = TD::Enum(EnumType {
        size: IntSize::U1,
        signed: false,
        members: vec![
            EnumMember { name: "red".to_owned(), value: 1 },
            EnumMember { name: "green".to_owned(), value: 2 },
        ],
    });
    let file = new_in_memory_file()?;
    for (i, desc) in [compound, enumeration, TD::VarLenUnicode].iter().enumerate() {
        let name = format!("t{i}");
        file.commit_datatype(&name, &Datatype::from_descriptor(desc)?)?;
        let committed = file.committed_datatype(&name)?;
        assert!(committed.as_datatype().is_committed());
        assert_eq!(committed.as_datatype().to_descriptor()?, *desc);
    }
    Ok(())
}

#[test]
pub fn test_eq() {
    assert_eq!(Datatype::from_type::<u32>().unwrap(), Datatype::from_type::<u32>().unwrap());
    assert_ne!(Datatype::from_type::<u16>().unwrap(), Datatype::from_type::<u32>().unwrap());
}

#[test]
fn test_print_display_debug_datatype_bool() {
    let dt = Datatype::from_type::<bool>().unwrap();

    assert_str_eq!(format!("{dt}"), "bool");
    assert_str_eq!(format!("{dt:?}"), "<HDF5 datatype: bool>");
    assert_str_eq!(format!("{dt:#?}"), "<HDF5 datatype: bool>");
}

#[test]
fn test_print_display_debug_datatype_f64() {
    let dt = Datatype::from_type::<f64>().unwrap();

    assert_str_eq!(format!("{dt}"), "float64");
    assert_str_eq!(format!("{dt:?}"), "<HDF5 datatype: float64>");
    assert_str_eq!(format!("{dt:#?}"), "<HDF5 datatype: float64>");
}

#[test]
fn test_print_display_debug_datatype_color_enum() {
    #[allow(dead_code)] // "we use the type, we just don't construct it"
    #[derive(H5Type)]
    #[repr(u8)]
    enum Color {
        R = 1,
        G = 2,
        B = 3,
    }
    let dt = Datatype::from_type::<Color>().unwrap();

    assert_eq!(
        dt.to_descriptor().unwrap(),
        TD::Enum(EnumType {
            size: IntSize::U1,
            signed: false,
            members: vec![
                EnumMember { name: "R".into(), value: 1 },
                EnumMember { name: "G".into(), value: 2 },
                EnumMember { name: "B".into(), value: 3 }
            ]
        })
    );

    assert_str_eq!(format!("{dt}"), "enum (uint8)");
    assert_str_eq!(format!("{dt:?}"), "<HDF5 datatype: enum (uint8)>");
    assert_str_eq!(format!("{dt:#?}"), "<HDF5 datatype: enum (uint8)>");
}

#[test]
fn test_print_display_debug_datatype_var_len_unicode() {
    let dt = Datatype::from_type::<VarLenUnicode>().unwrap();
    assert!(dt.is::<VarLenUnicode>());

    assert_eq!(dt.to_descriptor().unwrap(), TD::VarLenUnicode);

    assert_str_eq!(format!("{dt}"), "unicode (var len)");
    assert_str_eq!(format!("{dt:?}"), "<HDF5 datatype: unicode (var len)>");
    assert_str_eq!(format!("{dt:#?}"), "<HDF5 datatype: unicode (var len)>");
}

#[test]
fn test_print_display_debug_datatype_fixed_len_unicode() {
    const SIZE: usize = 10;
    let dt = Datatype::from_type::<FixedUnicode<SIZE>>().unwrap();
    assert!(dt.is::<FixedUnicode<SIZE>>());

    assert_eq!(dt.to_descriptor().unwrap(), TD::FixedUnicode(SIZE));

    assert_str_eq!(format!("{dt}"), "unicode (len 10)");
    assert_str_eq!(format!("{dt:?}"), "<HDF5 datatype: unicode (len 10)>");
    assert_str_eq!(format!("{dt:#?}"), "<HDF5 datatype: unicode (len 10)>");
}

#[test]
pub fn test_string_padding_fixed_ascii() -> hdf5::Result<()> {
    let datatype = Datatype::from_type::<FixedAscii<8>>()?;

    assert_eq!(datatype.string_padding()?, StringPadding::NullPadded);
    Ok(())
}

#[test]
pub fn test_string_padding_varlen_unicode() -> hdf5::Result<()> {
    let datatype = Datatype::from_type::<VarLenUnicode>()?;

    assert_eq!(datatype.string_padding()?, StringPadding::NullTerminated);
    Ok(())
}

#[test]
pub fn test_string_padding_rejects_integer_datatype() -> hdf5::Result<()> {
    let integer = Datatype::from_type::<u16>()?;

    let error = integer.string_padding().unwrap_err();

    let hdf5::Error::Internal(message) = error else {
        panic!("Expected a datatype class error, got {error:?}");
    };
    assert_str_eq!(message, "Expected a string datatype");
    Ok(())
}

#[test]
pub fn test_with_string_padding_fixed_ascii() -> hdf5::Result<()> {
    let original = Datatype::from_type::<FixedAscii<8>>()?;

    let configured = original.with_string_padding(StringPadding::SpacePadded)?;

    assert_eq!(configured.string_padding()?, StringPadding::SpacePadded);
    Ok(())
}

#[test]
pub fn test_with_string_padding_varlen_unicode() -> hdf5::Result<()> {
    let original = Datatype::from_type::<VarLenUnicode>()?;

    let configured = original.with_string_padding(StringPadding::NullPadded)?;

    assert_eq!(configured.string_padding()?, StringPadding::NullPadded);
    Ok(())
}

#[test]
pub fn test_with_string_padding_does_not_modify_original() -> hdf5::Result<()> {
    let original = Datatype::from_type::<FixedAscii<8>>()?;

    let configured = original.with_string_padding(StringPadding::SpacePadded)?;

    assert_eq!(original.string_padding()?, StringPadding::NullPadded);
    assert_eq!(configured.to_descriptor()?, TD::FixedAscii(8));
    Ok(())
}

#[test]
pub fn test_with_string_padding_rejects_integer_datatype() -> hdf5::Result<()> {
    let integer = Datatype::from_type::<u16>()?;

    let error = integer.with_string_padding(StringPadding::SpacePadded).unwrap_err();

    let hdf5::Error::Internal(message) = error else {
        panic!("Expected a datatype class error, got {error:?}");
    };
    assert_str_eq!(message, "Expected a string datatype");
    Ok(())
}

#[test]
pub fn test_with_string_padding_committed_datatype() -> hdf5::Result<()> {
    let file = new_in_memory_file()?;
    let original = Datatype::from_type::<FixedAscii<8>>()?;
    file.commit_datatype("text", &original)?;
    let committed = file.committed_datatype("text")?;

    let configured = committed.as_datatype().with_string_padding(StringPadding::SpacePadded)?;

    assert!(!configured.is_committed());
    assert_eq!(configured.string_padding()?, StringPadding::SpacePadded);
    assert!(committed.as_datatype().is_committed());
    assert_eq!(committed.as_datatype().string_padding()?, StringPadding::NullPadded);
    Ok(())
}

#[test]
pub fn test_compound_members() -> hdf5::Result<()> {
    let compound = Datatype::from_descriptor(&TD::Compound(CompoundType {
        fields: vec![
            CompoundField::typed::<u16>("id", 0, 0),
            CompoundField::typed::<f64>("value", 8, 1),
        ],
        size: 16,
    }))?;

    let members = compound.compound_members()?;

    assert_eq!(
        members,
        vec![
            CompoundMember {
                name: "id".into(),
                offset: 0,
                datatype: Datatype::from_type::<u16>()?
            },
            CompoundMember {
                name: "value".into(),
                offset: 8,
                datatype: Datatype::from_type::<f64>()?,
            },
        ]
    );
    Ok(())
}

#[test]
pub fn test_compound_members_nested_compound() -> hdf5::Result<()> {
    let inner = TD::Compound(CompoundType {
        fields: vec![CompoundField::typed::<i8>("x", 0, 0), CompoundField::typed::<i8>("y", 1, 1)],
        size: 2,
    });
    let outer = Datatype::from_descriptor(&TD::Compound(CompoundType {
        fields: vec![CompoundField::new("point", inner, 0, 0)],
        size: 2,
    }))?;

    let members = outer.compound_members()?[0].datatype.compound_members()?;

    assert_eq!(
        members,
        vec![
            CompoundMember { name: "x".into(), offset: 0, datatype: Datatype::from_type::<i8>()? },
            CompoundMember { name: "y".into(), offset: 1, datatype: Datatype::from_type::<i8>()? },
        ]
    );
    Ok(())
}

#[test]
pub fn test_compound_members_string_member() -> hdf5::Result<()> {
    let compound = Datatype::from_descriptor(&TD::Compound(CompoundType {
        fields: vec![CompoundField::typed::<VarLenUnicode>("text", 0, 0)],
        size: VarLenUnicode::type_descriptor().size(),
    }))?;

    let members = compound.compound_members()?;

    assert_eq!(
        members,
        vec![CompoundMember {
            name: "text".into(),
            offset: 0,
            datatype: Datatype::from_type::<VarLenUnicode>()?,
        }]
    );
    Ok(())
}

#[test]
pub fn test_compound_members_empty() -> hdf5::Result<()> {
    let compound =
        Datatype::from_descriptor(&TD::Compound(CompoundType { fields: vec![], size: 4 }))?;

    let members = compound.compound_members()?;

    assert_eq!(members, Vec::new());
    Ok(())
}

#[test]
pub fn test_compound_members_committed_datatype() -> hdf5::Result<()> {
    let file = new_in_memory_file()?;
    let original = Datatype::from_descriptor(&TD::Compound(CompoundType {
        fields: vec![
            CompoundField::typed::<u16>("id", 0, 0),
            CompoundField::typed::<f64>("value", 8, 1),
        ],
        size: 16,
    }))?;
    file.commit_datatype("record", &original)?;
    let committed = file.committed_datatype("record")?;

    let members = committed.as_datatype().compound_members()?;

    assert_eq!(members, original.compound_members()?);
    Ok(())
}

#[test]
pub fn test_compound_members_rejects_integer_datatype() -> hdf5::Result<()> {
    let integer = Datatype::from_type::<u16>()?;

    let error = integer.compound_members().unwrap_err();

    let hdf5::Error::Internal(message) = error else {
        panic!("Expected a datatype class error, got {error:?}");
    };
    assert_str_eq!(message, "Expected a compound datatype");
    Ok(())
}
