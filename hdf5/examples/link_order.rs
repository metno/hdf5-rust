//! List the links of a group in the order they were created
//!
//! HDF5 keeps the links of a group in a name index. A group can also record the order in which its
//! links were created, which is what h5py does with `track_order=True` and what netCDF-4 does for
//! every group, so a reader sees the variables in the order the writer added them. Tracking has to
//! be requested when the group is created and cannot be turned on later.

use hdf5::plist::group_create::LinkCreationOrder;
use hdf5::{File, IndexType, IterationCursor, IterationOrder, LinkType, MajorErrorCode, Result};
use hdf5_metno as hdf5;

const FILE_NAME: &str = "link_order.h5";
const VARIABLES: [&str; 5] = ["time", "latitude", "longitude", "temperature", "pressure"];
const PAGE_SIZE: usize = 3;

fn write() -> Result<()> {
    let file = File::create(FILE_NAME)?;

    let tracked = file
        .create_group_builder()
        .with_gcpl(|gcpl| gcpl.link_creation_order(LinkCreationOrder::Tracked))
        .create("tracked")?;
    for name in VARIABLES {
        tracked.new_dataset::<f64>().shape(24).create(name)?;
    }
    tracked.link_soft("temperature", "temp")?;

    // A group created with the defaults has only the name index.
    let untracked = file.create_group("untracked")?;
    for name in VARIABLES {
        untracked.new_dataset::<f64>().shape(24).create(name)?;
    }
    Ok(())
}

fn read() -> Result<()> {
    let file = File::open(FILE_NAME)?;
    let tracked = file.group("tracked")?;

    // member_names() and iter_visit_default() walk the name index.
    println!("by name: {:?}", tracked.member_names()?);

    // The creation index returns the writer's order, and LinkInfo carries the position in it.
    let mut by_creation = vec![];
    tracked.iter_visit(IndexType::CreationOrder, IterationOrder::Increasing, |name, info| {
        let order = info.creation_order.expect("the group tracks link creation order");
        println!("created {order}: {name}");
        by_creation.push(name.to_owned());
        Ok(())
    })?;
    assert_eq!(by_creation, ["time", "latitude", "longitude", "temperature", "pressure", "temp"]);

    // find_link stops at the first link the closure returns a value for.
    let first_soft =
        tracked.find_link(IndexType::CreationOrder, IterationOrder::Increasing, |name, info| {
            if info.link_type == LinkType::Soft { Ok(Some(name.to_owned())) } else { Ok(None) }
        })?;
    println!("first soft link: {first_soft:?}");
    assert_eq!(first_soft, Some("temp".to_owned()));

    // A cursor resumes a stopped iteration, here to read the links in pages.
    let mut cursor = IterationCursor::start(IndexType::CreationOrder, IterationOrder::Increasing);
    let mut page = vec![];
    while let Some(((), next)) = tracked.iter_visit_from(cursor, |name, _| {
        page.push(name.to_owned());
        if page.len() == PAGE_SIZE { Ok(Some(())) } else { Ok(None) }
    })? {
        println!("page from {}: {page:?}", cursor.position());
        page.clear();
        cursor = next;
    }

    // Creation order is not available in a group that never tracked it.
    let err = file
        .group("untracked")?
        .iter_visit(IndexType::CreationOrder, IterationOrder::Increasing, |_, _| Ok(()))
        .expect_err("the group does not track link creation order");
    assert!(err.contains_major(MajorErrorCode::SymbolTable));
    println!("untracked group: {err}");
    Ok(())
}

fn main() -> Result<()> {
    write()?;
    read()
}
