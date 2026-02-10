use std::str::FromStr;

use hdf5::types::VarLenUnicode;
use hdf5_metno as hdf5;

mod common;

use self::common::util::new_in_memory_file;

#[test]
#[cfg(feature = "blosc")]
fn blosc_blosclz() -> Result<(), Box<dyn std::error::Error>> {
    let file = new_in_memory_file()?;

    let dset_name = "x";
    let n_samples = 10;
    let input_data = vec![VarLenUnicode::from_str("test").unwrap(); n_samples];

    let ds_builder = file
        .new_dataset::<VarLenUnicode>()
        .shape((n_samples,))
        .chunk((n_samples,))
        .blosc_blosclz(5, true);

    let ds = ds_builder.create(dset_name)?;
    ds.write(&input_data)?;

    let read: ndarray::Array1<VarLenUnicode> =
        file.dataset(dset_name)?.read::<VarLenUnicode, ndarray::Ix1>()?;
    assert_eq!(read.to_vec(), input_data, "read data must match written data");
    Ok(())
}

#[test]
#[cfg(feature = "blosc-lz4")]
fn blosc_lz4() -> Result<(), Box<dyn std::error::Error>> {
    let file = new_in_memory_file()?;

    let dset_name = "x";
    let n_samples = 10;
    let input_data = vec![VarLenUnicode::from_str("test")?; n_samples];
    let ds_builder = file
        .new_dataset::<VarLenUnicode>()
        .shape((n_samples,))
        .chunk((n_samples,))
        .blosc_lz4(5, true);

    let ds = ds_builder.create(dset_name)?;
    ds.write(&input_data)?;

    let read: ndarray::Array1<VarLenUnicode> =
        file.dataset(dset_name)?.read::<VarLenUnicode, ndarray::Ix1>()?;
    assert_eq!(read.to_vec(), input_data, "read data must match written data");
    Ok(())
}

#[test]
#[cfg(feature = "blosc-zlib")]
fn blosc_zlib() -> Result<(), Box<dyn std::error::Error>> {
    let file = new_in_memory_file()?;

    let dset_name = "x";
    let n_samples = 10;
    let input_data = vec![VarLenUnicode::from_str("test")?; n_samples];
    let ds_builder = file
        .new_dataset::<VarLenUnicode>()
        .shape((n_samples,))
        .chunk((n_samples,))
        .blosc_zlib(5, true);

    let ds = ds_builder.create(dset_name)?;
    ds.write(&input_data)?;

    let read: ndarray::Array1<VarLenUnicode> =
        file.dataset(dset_name)?.read::<VarLenUnicode, ndarray::Ix1>()?;
    assert_eq!(read.to_vec(), input_data, "read data must match written data");
    Ok(())
}

#[test]
#[cfg(feature = "lzf")]
fn lzf() -> Result<(), Box<dyn std::error::Error>> {
    let file = new_in_memory_file()?;

    let dset_name = "x";
    let n_samples = 10;
    let input_data = vec![VarLenUnicode::from_str("test")?; n_samples];
    let ds_builder =
        file.new_dataset::<VarLenUnicode>().shape((n_samples,)).chunk((n_samples,)).lzf();

    let ds = ds_builder.create(dset_name)?;
    ds.write(&input_data)?;

    let read: ndarray::Array1<VarLenUnicode> =
        file.dataset(dset_name)?.read::<VarLenUnicode, ndarray::Ix1>()?;
    assert_eq!(read.to_vec(), input_data, "read data must match written data");
    Ok(())
}

#[test]
fn szip() -> Result<(), Box<dyn std::error::Error>> {
    if !hdf5::filters::szip_available() {
        return Ok(());
    }
    let file = new_in_memory_file()?;

    let dset_name = "x";
    let n_samples = 10;
    let input_data = vec![VarLenUnicode::from_str("test")?; n_samples];
    let ds_builder = file
        .new_dataset::<VarLenUnicode>()
        .shape((n_samples,))
        .chunk((n_samples,))
        .szip(hdf5::filters::SZip::Entropy, 8);

    let ds = ds_builder.create(dset_name)?;
    ds.write(&input_data)?;

    let read: ndarray::Array1<VarLenUnicode> =
        file.dataset(dset_name)?.read::<VarLenUnicode, ndarray::Ix1>()?;
    assert_eq!(read.to_vec(), input_data, "read data must match written data");
    Ok(())
}

#[test]
#[cfg(feature = "zlib")]
fn deflate() -> Result<(), Box<dyn std::error::Error>> {
    let file = new_in_memory_file()?;

    let dset_name = "x";
    let n_samples = 10;
    let input_data = vec![VarLenUnicode::from_str("test")?; n_samples];
    let ds_builder =
        file.new_dataset::<VarLenUnicode>().shape((n_samples,)).chunk((n_samples,)).deflate(5);

    let ds = ds_builder.create(dset_name)?;
    ds.write(&input_data)?;

    let read: ndarray::Array1<VarLenUnicode> =
        file.dataset(dset_name)?.read::<VarLenUnicode, ndarray::Ix1>()?;
    assert_eq!(read.to_vec(), input_data, "read data must match written data");
    Ok(())
}
