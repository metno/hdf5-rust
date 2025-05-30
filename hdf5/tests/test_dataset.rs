use std::convert::TryFrom;
use std::fmt;
use std::io::{Read, Seek, SeekFrom};

use ndarray::{s, Array1, Array2, ArrayD, IxDyn, SliceInfo};
use rand::prelude::{Rng, SeedableRng, SmallRng};

use hdf5_metno as hdf5;
use hdf5_types::TypeDescriptor;

mod common;

use self::common::gen::{
    gen_arr, gen_slice, Enum, FixedStruct, Gen, RenameEnum, RenameStruct, RenameTupleStruct,
    TupleStruct, VarLenStruct,
};
use self::common::util::new_in_memory_file;

fn test_write_slice<T, R>(
    rng: &mut R, ds: &hdf5::Dataset, arr: &ArrayD<T>, default_value: &T, _ndim: usize,
) -> hdf5::Result<()>
where
    T: hdf5::H5Type + fmt::Debug + PartialEq + Gen + Clone,
    R: Rng + ?Sized,
{
    let shape = arr.shape();
    let slice = gen_slice(rng, shape);

    // Take a random slice of the dataset, and convert it to a standard dense layout
    let sliced_array_view = arr.slice(slice.as_ref());
    let mut sliced_array_copy = ArrayD::from_elem(sliced_array_view.shape(), default_value.clone());
    sliced_array_copy.assign(&sliced_array_view);

    // Write these elements into their 'correct' places in the matrix
    {
        let dsw = ds.as_writer();
        dsw.write_slice(&sliced_array_copy, slice.clone())?;
    }

    // Read back out the random from the full dataset
    let full_ds = ds.read_dyn::<T>()?;
    let read_slice = full_ds.slice(slice.as_ref());

    assert_eq!(sliced_array_view, read_slice);
    Ok(())
}

fn test_read_slice<T, R>(
    rng: &mut R, ds: &hdf5::Dataset, arr: &ArrayD<T>, ndim: usize,
) -> hdf5::Result<()>
where
    T: hdf5::H5Type + fmt::Debug + PartialEq + Gen,
    R: Rng + ?Sized,
{
    ds.write(arr)?;

    // Test various sliced reads
    let shape = arr.shape();

    let out_dyn = ds.read_dyn::<T>();
    assert_eq!(arr, &out_dyn?.into_dimensionality().unwrap());

    let dsr = ds.as_reader();

    for _ in 0..10 {
        let slice = gen_slice(rng, shape);

        // Do a sliced HDF5 read
        let sliced_read: ArrayD<T> = dsr.read_slice(slice.clone()).unwrap();

        // Slice the full dataset
        let sliced_dataset = arr.slice(slice.as_ref());

        // Ensure that the H5 sliced read matches the ndarray slice of the original array.
        if sliced_read != sliced_dataset {
            println!("{:?}", slice);
        }
        assert_eq!(sliced_read, sliced_dataset);
    }

    // Test that we get an error if we use the wrong dimensionality when slicing.
    let mut bad_shape = Vec::from(shape);
    bad_shape.push(1);
    let bad_slice = gen_slice(rng, &bad_shape);
    let bad_slice: SliceInfo<_, IxDyn, IxDyn> =
        ndarray::SliceInfo::try_from(bad_slice.as_slice()).unwrap();

    let bad_sliced_read: hdf5::Result<ArrayD<T>> = dsr.read_slice(bad_slice);
    assert!(bad_sliced_read.is_err());

    // Tests for dimension-dropping slices with static dimensionality.
    if ndim == 2 && shape[0] > 0 && shape[1] > 0 {
        let v: Array1<T> = dsr.read_slice_1d(s![0, ..])?;
        assert_eq!(shape[1], v.shape()[0]);

        let v: Array1<T> = dsr.read_slice_1d(s![.., 0])?;
        assert_eq!(shape[0], v.shape()[0]);
    }

    if ndim == 3 && shape[0] > 0 && shape[1] > 0 && shape[2] > 0 {
        let v: Array2<T> = dsr.read_slice_2d(s![0, .., ..])?;
        assert_eq!(shape[1], v.shape()[0]);
        assert_eq!(shape[2], v.shape()[1]);

        let v: Array1<T> = dsr.read_slice_1d(s![0, 0, ..])?;
        assert_eq!(shape[2], v.shape()[0]);
    }

    Ok(())
}

fn test_read<T>(ds: &hdf5::Dataset, arr: &ArrayD<T>, ndim: usize) -> hdf5::Result<()>
where
    T: hdf5::H5Type + fmt::Debug + PartialEq + Gen,
{
    ds.write(arr)?;

    // read_raw()
    let out_vec = ds.read_raw::<T>();
    assert_eq!(arr.as_slice().unwrap(), out_vec?.as_slice());

    // read_dyn()
    let out_dyn = ds.read_dyn::<T>();
    assert_eq!(arr, &out_dyn?.into_dimensionality().unwrap());

    // read_scalar()
    let out_scalar = ds.read_scalar::<T>();
    if ndim == 0 {
        assert_eq!(arr.as_slice().unwrap()[0], out_scalar?);
    } else {
        assert!(out_scalar.is_err());
    }

    // read_1d()
    let out_1d = ds.read_1d::<T>();
    if ndim == 1 {
        assert_eq!(arr, &out_1d?.into_dimensionality().unwrap());
    } else {
        assert!(out_1d.is_err());
    }

    // read_2d()
    let out_2d = ds.read_2d::<T>();
    if ndim == 2 {
        assert_eq!(arr, &out_2d?.into_dimensionality().unwrap());
    } else {
        assert!(out_2d.is_err());
    }

    Ok(())
}

fn test_write<T>(ds: &hdf5::Dataset, arr: &ArrayD<T>, ndim: usize) -> hdf5::Result<()>
where
    T: hdf5::H5Type + fmt::Debug + PartialEq + Gen,
{
    // .write()
    ds.write(arr)?;
    assert_eq!(&ds.read_dyn::<T>()?, arr);

    // .write_scalar()
    if ndim == 0 {
        ds.write_scalar(&arr.as_slice().unwrap()[0])?;
        assert_eq!(&ds.read_dyn::<T>()?, arr);
    } else if arr.len() > 0 {
        assert!(ds.write_scalar(&arr.as_slice().unwrap()[0]).is_err());
    }

    // .write_raw()
    ds.write_raw(arr.as_slice().unwrap())?;
    assert_eq!(&ds.read_dyn::<T>()?, arr);

    Ok(())
}

fn test_byte_read_seek_impl(ds: &hdf5::Dataset, arr: &ArrayD<u8>, ndim: usize) -> hdf5::Result<()> {
    let mut rng = SmallRng::seed_from_u64(42);
    ds.write(arr)?;

    // Read whole
    let reader = ds.as_byte_reader();
    let mut reader = if ndim != 1 {
        assert!(reader.is_err());
        return Ok(());
    } else {
        reader.unwrap()
    };
    let mut out_bytes = vec![0u8; arr.len()];
    reader.read(&mut out_bytes.as_mut_slice()).expect("io::Read failed");
    assert_eq!(out_bytes.as_slice(), arr.as_slice().unwrap());

    // Read in chunks
    let mut reader = reader.clone();
    reader.seek(std::io::SeekFrom::Start(0)).expect("io::Seek failed");
    let mut pos = 0;
    while pos < arr.len() {
        let chunk_len: usize = rng.random_range(1..arr.len() + 1);
        let mut chunk = vec![0u8; chunk_len];
        let n_read = reader.read(&mut chunk).expect("io::Read failed");
        if pos + chunk_len < arr.len() {
            // We did not read until end. Thus, the chunk should be fully filled.
            assert_eq!(chunk_len, n_read);
        }
        assert_eq!(&chunk[..n_read], arr.slice(s![pos..pos + n_read]).as_slice().unwrap());
        pos += chunk_len;
    }

    // Seek to the beginning and read again
    reader.seek(SeekFrom::Start(0)).expect("io::Seek failed");
    let mut out_bytes = vec![0u8; arr.len()];
    reader.read(&mut out_bytes.as_mut_slice()).expect("io::Read failed");
    assert_eq!(out_bytes.as_slice(), arr.as_slice().unwrap());

    // Seek to a random position from start
    let pos = rng.random_range(0..arr.len() + 1) as u64;
    let seek_pos = reader.seek(SeekFrom::Start(pos)).expect("io::Seek failed") as usize;
    let mut out_bytes = vec![0u8; arr.len() - seek_pos];
    reader.read(&mut out_bytes.as_mut_slice()).expect("io::Read failed");
    assert_eq!(out_bytes.as_slice(), arr.slice(s![seek_pos..]).as_slice().unwrap());

    // Seek from current position
    let orig_pos = reader.seek(SeekFrom::Start(pos)).expect("io::Seek failed") as i64;
    let rel_pos = rng.random_range(-(orig_pos as i64)..(arr.len() as i64 - orig_pos) + 1);
    let pos_res = reader.seek(SeekFrom::Current(rel_pos));
    if (rel_pos + orig_pos) < 0 {
        assert!(pos_res.is_err()) // We cannot seek before start
    } else {
        let seek_pos = pos_res.unwrap() as usize;
        assert_eq!(rel_pos + orig_pos, seek_pos as i64);
        let mut out_bytes = vec![0u8; arr.len() - seek_pos];
        reader.read(&mut out_bytes.as_mut_slice()).expect("io::Read failed");
        assert_eq!(out_bytes.as_slice(), arr.slice(s![seek_pos..]).as_slice().unwrap());
    }

    // Seek to a random position from end
    let pos = -(rng.random_range(0..arr.len() + 1) as i64);
    let seek_pos = reader.seek(SeekFrom::End(pos)).expect("io::Seek failed") as usize;
    assert_eq!(pos, seek_pos as i64 - arr.len() as i64);
    let mut out_bytes = vec![0u8; arr.len() - seek_pos];
    reader.read(&mut out_bytes.as_mut_slice()).expect("io::Read failed");
    assert_eq!(out_bytes.as_slice(), arr.slice(s![seek_pos..]).as_slice().unwrap());

    // Seek before start
    assert!(reader.seek(SeekFrom::End(-(arr.len() as i64) - 1)).is_err());

    // Test stream position start
    // Requires Rust 1.55.0: reader.rewind().expect("io::Seek::rewind failed");
    assert_eq!(0, reader.seek(SeekFrom::Start(0)).unwrap());
    assert_eq!(0, reader.stream_position().unwrap());
    assert_eq!(0, reader.seek(SeekFrom::End(-(arr.len() as i64))).unwrap());
    assert_eq!(0, reader.stream_position().unwrap());
    Ok(())
}

fn test_read_write<T>() -> hdf5::Result<()>
where
    T: hdf5::H5Type + fmt::Debug + PartialEq + Gen + Clone,
{
    let td = T::type_descriptor();
    let mut packed = vec![false];
    if let TypeDescriptor::Compound(_) = td {
        packed.push(true);
    }

    let mut rng = SmallRng::seed_from_u64(42);
    let file = new_in_memory_file()?;

    for packed in &packed {
        for ndim in 0..=4 {
            for _ in 0..=20 {
                for mode in 0..4 {
                    let arr: ArrayD<T> = gen_arr(&mut rng, ndim);

                    let ds: hdf5::Dataset =
                        file.new_dataset::<T>().packed(*packed).shape(arr.shape()).create("x")?;
                    let ds = scopeguard::guard(ds, |ds| {
                        drop(ds);
                        drop(file.unlink("x"));
                    });

                    if mode == 0 {
                        test_read(&ds, &arr, ndim)?;
                    } else if mode == 1 {
                        test_write(&ds, &arr, ndim)?;
                    } else if mode == 2 {
                        test_read_slice(&mut rng, &ds, &arr, ndim)?;
                    } else if mode == 3 {
                        let default_value = T::random(&mut rng);
                        test_write_slice(&mut rng, &ds, &arr, &default_value, ndim)?;
                    }
                }
            }
        }
    }

    Ok(())
}

#[test]
fn test_read_write_primitive() -> hdf5::Result<()> {
    test_read_write::<i8>()?;
    test_read_write::<i16>()?;
    test_read_write::<i32>()?;
    test_read_write::<i64>()?;
    test_read_write::<u8>()?;
    test_read_write::<u16>()?;
    test_read_write::<u32>()?;
    test_read_write::<u64>()?;
    test_read_write::<bool>()?;
    test_read_write::<f32>()?;
    test_read_write::<f64>()?;
    Ok(())
}

#[cfg(feature = "f16")]
#[test]
fn test_read_write_f16() -> hdf5::Result<()> {
    test_read_write::<::half::f16>()?;
    Ok(())
}

#[cfg(feature = "complex")]
#[test]
fn test_read_write_complex() -> hdf5::Result<()> {
    test_read_write::<::num_complex::Complex32>()?;
    test_read_write::<::num_complex::Complex64>()?;
    Ok(())
}

#[test]
fn test_read_write_enum() -> hdf5::Result<()> {
    test_read_write::<Enum>()
}

#[test]
fn test_read_write_tuple_struct() -> hdf5::Result<()> {
    test_read_write::<TupleStruct>()
}

#[test]
fn test_read_write_fixed_struct() -> hdf5::Result<()> {
    test_read_write::<FixedStruct>()
}

#[test]
fn test_read_write_varlen_struct() -> hdf5::Result<()> {
    test_read_write::<VarLenStruct>()
}

#[test]
fn test_create_on_databuilder() {
    let file = new_in_memory_file().unwrap();

    let _ds = file.new_dataset_builder().empty::<i32>().create("ds1").unwrap();
    let _ds = file.new_dataset_builder().with_data(&[1_i32, 2, 3]).create("ds2").unwrap();
    let _ds = file.new_dataset::<i32>().create("ds3").unwrap();
    let _ds = file.new_dataset::<i32>().shape(2).create("ds4").unwrap();
}

#[test]
#[cfg(feature = "have-filter-deflate")]
fn test_issue_223() {
    let file = new_in_memory_file().unwrap();

    let data: &[u16] = &[];
    let _ds = file.new_dataset_builder().deflate(3).with_data(data).create("ds2").unwrap();

    let data: Array2<u16> = Array2::from_shape_fn((0, 0), |(_i, _j)| unreachable!());
    let _ds = file.new_dataset_builder().deflate(3).with_data(&data).create("ds3").unwrap();
}

#[test]
fn test_read_write_rename_fields() -> hdf5::Result<()> {
    test_read_write::<RenameStruct>()?;
    test_read_write::<RenameTupleStruct>()?;
    test_read_write::<RenameEnum>()?;
    Ok(())
}

#[test]
fn test_byte_read_seek() -> hdf5::Result<()> {
    let mut rng = SmallRng::seed_from_u64(42);
    let file = new_in_memory_file()?;

    for ndim in 0..=2 {
        for _ in 0..=20 {
            let arr: ArrayD<u8> = gen_arr(&mut rng, ndim);

            let ds: hdf5::Dataset = file.new_dataset::<u8>().shape(arr.shape()).create("x")?;
            let ds = scopeguard::guard(ds, |ds| {
                drop(ds);
                drop(file.unlink("x"));
            });

            test_byte_read_seek_impl(&ds, &arr, ndim)?;
        }
    }
    Ok(())
}

#[test]
#[cfg(feature = "blosc")]
fn test_blosc_filters_write_compressed() {
    let file = new_in_memory_file().unwrap();
    let data = [1i32; 1024];
    let size = (std::mem::size_of::<i32>() * data.len()) as u64;

    // Sanity Check / Baseline test
    let _ds = file.new_dataset_builder().with_data(&data).create("raw").unwrap();
    assert!(_ds.storage_size() == size);
    assert!(_ds.filters().is_empty());

    let _ds = file
        .new_dataset_builder()
        .with_data(&data)
        .blosc_blosclz(9, true)
        .create("blosclz")
        .unwrap();
    assert!(_ds.storage_size() < size);
    assert_eq!(_ds.filters(), vec![hdf5::filters::Filter::blosc_blosclz(9, true)]);

    #[cfg(feature = "blosc-lz4")]
    {
        let _ds =
            file.new_dataset_builder().with_data(&data).blosc_lz4(9, true).create("lz4").unwrap();
        assert!(_ds.storage_size() < size);
        assert_eq!(_ds.filters(), vec![hdf5::filters::Filter::blosc_lz4(9, true)]);

        let _ds = file
            .new_dataset_builder()
            .with_data(&data)
            .blosc_lz4hc(9, true)
            .create("lz4hc")
            .unwrap();
        assert!(_ds.storage_size() < size);
        assert_eq!(_ds.filters(), vec![hdf5::filters::Filter::blosc_lz4hc(9, true)]);
    }

    #[cfg(feature = "blosc-zlib")]
    {
        let _ds =
            file.new_dataset_builder().with_data(&data).blosc_zlib(9, true).create("zlib").unwrap();
        assert!(_ds.storage_size() < size);
        assert_eq!(_ds.filters(), vec![hdf5::filters::Filter::blosc_zlib(9, true)]);
    }

    #[cfg(feature = "blosc-zstd")]
    {
        let _ds =
            file.new_dataset_builder().with_data(&data).blosc_zstd(9, true).create("zstd").unwrap();
        assert!(_ds.storage_size() < size);
        assert_eq!(_ds.filters(), vec![hdf5::filters::Filter::blosc_zstd(9, true)]);
    }

    #[cfg(feature = "blosc-snappy")]
    {
        let _ds = file
            .new_dataset_builder()
            .with_data(&data)
            .blosc_snappy(9, true)
            .create("snappy")
            .unwrap();
        assert!(_ds.storage_size() < size);
        assert_eq!(_ds.filters(), vec![hdf5::filters::Filter::blosc_snappy(9, true)]);
    }
}

#[test]
fn remove_attr() {
    let file = new_in_memory_file().unwrap();

    file.new_attr::<i32>().create("foo").unwrap();
    assert!(file.attr("foo").is_ok());
    file.delete_attr("foo").unwrap();
    assert!(file.attr("foo").is_err());

    let ds = file.new_dataset::<u8>().create("ds").unwrap();
    ds.new_attr::<i8>().create("bar").unwrap();
    assert!(ds.attr("bar").is_ok());
    ds.delete_attr("bar").unwrap();
    assert!(ds.attr("bar").is_err());
}
