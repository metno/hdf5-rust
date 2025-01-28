use super::gen::gen_ascii;

use hdf5_metno as hdf5;

pub fn random_filename() -> String {
    gen_ascii(&mut rand::rng(), 8)
}

pub fn new_in_memory_file() -> hdf5::Result<hdf5::File> {
    let filename = random_filename();
    hdf5::File::with_options().with_fapl(|p| p.core_filebacked(false)).create(&filename)
}
