//! List the links and attributes of a group in the order they were created
//!
//! HDF5 keeps the links of a group and the attributes of an object in a name index. Both can also
//! record the order in which they were created, which is what h5py does with `track_order=True`
//! and what netCDF-4 does for every group, so a reader sees variables and attributes in the order
//! the writer added them. Tracking has to be requested when the object is created and cannot be
//! turned on later.

use hdf5::plist::group_create::{AttrCreationOrder, LinkCreationOrder};
use hdf5::{File, IndexType, IterationCursor, IterationOrder, LinkType, MajorErrorCode, Result};
use hdf5_metno as hdf5;

const FILE_NAME: &str = "creation_order.h5";
const VARIABLES: [&str; 5] = ["time", "latitude", "longitude", "temperature", "pressure"];
const ATTRIBUTES: [&str; 3] = ["title", "history", "Conventions"];
const PAGE_SIZE: usize = 3;

fn write() -> Result<()> {
    let file = File::create(FILE_NAME)?;

    let tracked = file
        .create_group_builder()
        .with_gcpl(|gcpl| {
            gcpl.link_creation_order(LinkCreationOrder::Tracked)
                .attr_creation_order(AttrCreationOrder::Tracked)
        })
        .create("tracked")?;
    for name in VARIABLES {
        tracked.new_dataset::<f64>().shape(24).create(name)?;
    }
    tracked.link_soft("temperature", "temp")?;
    for name in ATTRIBUTES {
        tracked.new_attr::<f64>().create(name)?;
    }

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

    // Attributes follow the same index types. The tracked group lists them in creation order.
    println!("attributes by name: {:?}", tracked.attr_names()?);
    let mut attrs_by_creation = vec![];
    tracked.iter_attrs(IndexType::CreationOrder, IterationOrder::Increasing, |name, info| {
        let order = info.creation_order.expect("the group tracks attribute creation order");
        println!("created {order}: @{name}");
        attrs_by_creation.push(name.to_owned());
        Ok(())
    })?;
    assert_eq!(attrs_by_creation, ATTRIBUTES);

    // An attribute is opened by its position in either index, and its info looked up by name.
    let first = tracked.attr_by_index(IndexType::CreationOrder, IterationOrder::Increasing, 0)?;
    assert_eq!(first.name(), "title");
    let history = tracked.attr_info("history")?;
    println!("history: {history:?}");
    assert_eq!(history.creation_order, Some(1));

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
