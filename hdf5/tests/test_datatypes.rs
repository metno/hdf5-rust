#[macro_use]
mod common;

use hdf5::types::{TypeDescriptor as TD, *};
use hdf5::{from_id, Datatype, H5Type};
use hdf5_metno as hdf5;
use hdf5_sys::h5i::H5I_INVALID_HID;
use pretty_assertions::{assert_eq, assert_str_eq};

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
pub fn test_invalid_datatype() {
    assert_err!(from_id::<Datatype>(H5I_INVALID_HID), "Invalid handle id");
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
