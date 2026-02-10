use std::convert::TryFrom;
use std::fmt::{self, Debug};
use std::iter;

use hdf5::H5Type;
use hdf5::types::{FixedAscii, FixedUnicode, VarLenArray, VarLenAscii, VarLenUnicode};
use hdf5_metno as hdf5;

use half::f16;
use ndarray::{ArrayD, SliceInfo, SliceInfoElem};
use num_complex::Complex;
use rand::distr::StandardUniform;
use rand::distr::{Alphanumeric, Uniform};
use rand::prelude::{Distribution, IndexedRandom};
use rand::prelude::{Rng, RngExt};

pub fn gen_shape<R: Rng + ?Sized>(rng: &mut R, ndim: usize) -> Vec<usize> {
    iter::repeat(()).map(|_| rng.random_range(0..11)).take(ndim).collect()
}

pub fn gen_ascii<R: Rng + ?Sized>(rng: &mut R, len: usize) -> String {
    iter::repeat(()).map(|_| rng.sample(Alphanumeric)).map(char::from).take(len).collect()
}

/// Generate a random slice of elements inside the given `shape` dimension.
pub fn gen_slice<R: Rng + ?Sized>(
    rng: &mut R, shape: &[usize],
) -> SliceInfo<Vec<SliceInfoElem>, ndarray::IxDyn, ndarray::IxDyn> {
    let rand_slice: Vec<SliceInfoElem> =
        shape.into_iter().map(|s| gen_slice_one_dim(rng, *s)).collect();
    SliceInfo::try_from(rand_slice).unwrap()
}

/// Generate a random 1D slice of the interval [0, shape).
fn gen_slice_one_dim<R: Rng + ?Sized>(rng: &mut R, shape: usize) -> ndarray::SliceInfoElem {
    if shape == 0 {
        return ndarray::SliceInfoElem::Slice { start: 0, end: None, step: 1 };
    }

    if rng.random_bool(0.1) {
        ndarray::SliceInfoElem::Index(rng.random_range(0..shape) as isize)
    } else {
        let start = rng.random_range(0..shape) as isize;

        let end = if rng.random_bool(0.5) {
            None
        } else if rng.random_bool(0.9) {
            Some(rng.random_range(start as i64..shape as i64))
        } else {
            // Occasionally generate a slice with end < start.
            Some(rng.random_range(0..shape as i64))
        };

        let step =
            if rng.random_bool(0.9) { 1isize } else { rng.random_range(1..shape * 2) as isize };

        ndarray::SliceInfoElem::Slice { start, end: end.map(|x| x as isize), step }
    }
}

pub trait Gen: Sized + fmt::Debug {
    fn random<R: Rng + ?Sized>(rng: &mut R) -> Self;
}

macro_rules! impl_gen_primitive {
    ($ty:ty) => {
        impl Gen for $ty {
            fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
                rng.random()
            }
        }
    };
    ($ty:ty, $($tys:ty),+) => {
        impl_gen_primitive!($ty);
        impl_gen_primitive!($($tys),*);
    };
}

impl_gen_primitive!(u8, u16, u32, u64, i8, i16, i32, i64, bool, f32, f64);

macro_rules! impl_gen_tuple {
    ($t:ident) => (
        impl<$t> Gen for ($t,) where $t: Gen {
            fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
                (<$t as Gen>::random(rng),)
            }
        }
    );

    ($t:ident, $($tt:ident),*) => (
        impl<$t, $($tt),*> Gen for ($t, $($tt),*) where $t: Gen, $($tt: Gen),* {
            fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
                (<$t as Gen>::random(rng), $(<$tt as Gen>::random(rng)),*)
            }
        }
        impl_gen_tuple!($($tt),*);
    );
}

impl_gen_tuple! { A, B, C, D, E, F, G, H, I, J, K, L }

impl Gen for f16 {
    fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Self::from_f32(rng.random())
    }
}

impl<T: Debug> Gen for Complex<T>
where
    StandardUniform: Distribution<T>,
{
    fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Self::new(rng.random(), rng.random())
    }
}

pub fn gen_vec<R: Rng + ?Sized, T: Gen>(rng: &mut R, size: usize) -> Vec<T> {
    iter::repeat(()).map(|_| T::random(rng)).take(size).collect()
}

pub fn gen_arr<T, R>(rng: &mut R, ndim: usize) -> ArrayD<T>
where
    T: H5Type + Gen,
    R: Rng + ?Sized,
{
    let shape = gen_shape(rng, ndim);
    let size = shape.iter().product();
    let vec = gen_vec(rng, size);
    ArrayD::from_shape_vec(shape, vec).unwrap()
}

impl<const N: usize> Gen for FixedAscii<N> {
    fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let len = rng.sample(Uniform::new_inclusive(0, N).unwrap());
        let dist = Uniform::new_inclusive(0, 127).unwrap();
        let mut v = Vec::with_capacity(len);
        for _ in 0..len {
            v.push(rng.sample(dist));
        }
        unsafe { FixedAscii::from_ascii_unchecked(&v) }
    }
}

impl<const N: usize> Gen for FixedUnicode<N> {
    fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let len = rng.sample(Uniform::new_inclusive(0, N).unwrap());
        let mut s = String::new();
        for _ in 0..len {
            let c = rng.random::<char>();
            if c != '\0' {
                if s.as_bytes().len() + c.len_utf8() >= len {
                    break;
                }
                s.push(c);
            }
        }
        unsafe { FixedUnicode::from_str_unchecked(s) }
    }
}

impl Gen for VarLenAscii {
    fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let len = rng.sample(Uniform::new_inclusive(0, 8).unwrap());
        let dist = Uniform::new_inclusive(0, 127).unwrap();
        let mut v = Vec::with_capacity(len);
        for _ in 0..len {
            v.push(rng.sample(dist));
        }
        unsafe { VarLenAscii::from_ascii_unchecked(&v) }
    }
}

impl Gen for VarLenUnicode {
    fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let len = rng.sample(Uniform::new_inclusive(0, 8).unwrap());
        let mut s = String::new();
        while s.len() < len {
            let c = rng.random::<char>();
            if c != '\0' {
                s.push(c);
            }
        }
        unsafe { VarLenUnicode::from_str_unchecked(s) }
    }
}

impl<T: Gen + Copy> Gen for VarLenArray<T> {
    fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let len = rng.sample(Uniform::new_inclusive(0, 8).unwrap());
        let mut v = Vec::with_capacity(len);
        for _ in 0..len {
            v.push(Gen::random(rng));
        }
        VarLenArray::from_slice(&v)
    }
}

#[derive(H5Type, Clone, Copy, Debug, PartialEq)]
#[repr(i16)]
pub enum Enum {
    X = -2,
    Y = 3,
}

impl Gen for Enum {
    fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        *[Enum::X, Enum::Y].choose(rng).unwrap()
    }
}

#[derive(H5Type, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct TupleStruct(bool, Enum);

impl Gen for TupleStruct {
    fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        TupleStruct(Gen::random(rng), Gen::random(rng))
    }
}

#[derive(H5Type, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct FixedStruct {
    fa: FixedAscii<3>,
    fu: FixedUnicode<11>,
    array: [TupleStruct; 2],
}

impl Gen for FixedStruct {
    fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        FixedStruct {
            fa: Gen::random(rng),
            fu: Gen::random(rng),
            array: [Gen::random(rng), Gen::random(rng)],
        }
    }
}

#[derive(H5Type, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct VarLenStruct {
    va: VarLenAscii,
    vu: VarLenUnicode,
    vla: VarLenArray<Enum>,
}

impl Gen for VarLenStruct {
    fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        VarLenStruct { va: Gen::random(rng), vu: Gen::random(rng), vla: Gen::random(rng) }
    }
}

#[derive(H5Type, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct RenameStruct {
    first: i32,
    #[hdf5(rename = "field.second")]
    second: i64,
}

impl Gen for RenameStruct {
    fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        RenameStruct { first: Gen::random(rng), second: Gen::random(rng) }
    }
}

#[derive(H5Type, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct RenameTupleStruct(#[hdf5(rename = "my_boolean")] bool, #[hdf5(rename = "my_enum")] Enum);

impl Gen for RenameTupleStruct {
    fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        RenameTupleStruct(Gen::random(rng), Gen::random(rng))
    }
}

#[derive(H5Type, Clone, Copy, Debug, PartialEq)]
#[repr(i16)]
pub enum RenameEnum {
    #[hdf5(rename = "coord.x")]
    X = -2,
    #[hdf5(rename = "coord.y")]
    Y = 3,
}

impl Gen for RenameEnum {
    fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        *[RenameEnum::X, RenameEnum::Y].choose(rng).unwrap()
    }
}
