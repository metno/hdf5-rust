//! Share one record type across many datasets with a committed datatype
//!
//! A datatype built from a Rust type is transient: every dataset created from it stores its own
//! copy of the type. A committed datatype is an object in the file. Datasets created from it through
//! `DatasetBuilder::empty_as` or `with_data_as` refer to that object instead of copying it, so the record
//! layout is described once, can be documented with attributes, and can be looked up by name.

use hdf5::types::VarLenUnicode;
use hdf5::{Datatype, File, H5Type, Result};
use hdf5_metno as hdf5;

#[derive(H5Type, Clone, Copy, Debug, PartialEq)]
#[repr(i8)]
enum Quality {
    Good = 0,
    Suspect = 1,
    Missing = 2,
}

#[derive(H5Type, Clone, Debug, PartialEq)]
#[repr(C)]
struct Sample {
    time: f64,
    value: f32,
    quality: Quality,
}

const FILE_NAME: &str = "committed_datatype.h5";
const SCHEMA: &str = "types/sample";

fn samples(offset: f32) -> Vec<Sample> {
    (0..4)
        .map(|i| Sample {
            time: 60.0 * f64::from(i),
            value: offset + i as f32,
            quality: match i {
                2 => Quality::Suspect,
                3 => Quality::Missing,
                _ => Quality::Good,
            },
        })
        .collect()
}

fn write() -> Result<()> {
    let file = File::create(FILE_NAME)?;

    // Commit the record type once, at a path readers can look up.
    let sample = Datatype::from_type::<Sample>()?;
    file.create_group("types")?;
    file.commit_datatype(SCHEMA, &sample)?;

    // A committed datatype is an object, so it can carry attributes that document the layout.
    let schema = file.committed_datatype(SCHEMA)?;
    schema.new_attr::<u32>().create("schema_version")?.write_scalar(&1)?;
    let unit: VarLenUnicode = "seconds since 2024-01-01T00:00:00Z".parse().unwrap();
    schema.new_attr::<VarLenUnicode>().create("time_unit")?.write_scalar(&unit)?;

    // Every station dataset refers to the committed type instead of storing a copy.
    for (i, station) in ["oslo", "bergen"].iter().enumerate() {
        let data = samples(10.0 * i as f32);
        file.new_dataset_builder()
            .with_data_as(&data, &sample)
            .create(format!("stations/{station}").as_str())?;
    }

    // For contrast, a dataset built from the Rust type alone gets a transient copy of the type.
    let copy = file.new_dataset::<Sample>().shape(1).create("scratch")?;
    assert!(!copy.dtype()?.is_committed());
    Ok(())
}

fn append_station() -> Result<()> {
    let file = File::open_rw(FILE_NAME)?;

    // The type of an existing dataset is the committed type, so it creates more of the same.
    let template = file.dataset("stations/oslo")?.dtype()?;
    assert!(template.is_committed());
    let data = samples(20.0);
    file.new_dataset_builder()
        .empty_as(&template)
        .shape(data.len())
        .create("stations/tromso")?
        .write(&data)?;
    Ok(())
}

fn read() -> Result<()> {
    let file = File::open(FILE_NAME)?;

    // The schema and its documentation are found by name, without opening any dataset.
    let schema = file.committed_datatype(SCHEMA)?;
    let version = schema.attr("schema_version")?.read_scalar::<u32>()?;
    let unit = schema.attr("time_unit")?.read_scalar::<VarLenUnicode>()?;
    println!("schema {} version {version}, time in {unit}", schema.name());
    println!("layout: {:?}", schema.as_datatype().to_descriptor()?);

    // Every dataset created from the schema reports the committed type.
    let stations = file.group("stations")?;
    for name in stations.member_names()? {
        let ds = stations.dataset(&name)?;
        assert!(ds.dtype()?.is_committed());
        let data = ds.read_1d::<Sample>()?;
        let suspect = data.iter().filter(|s| s.quality != Quality::Good).count();
        println!("{name}: {} samples, {suspect} flagged", data.len());
    }
    Ok(())
}

fn main() -> Result<()> {
    write()?;
    append_station()?;
    read()
}
