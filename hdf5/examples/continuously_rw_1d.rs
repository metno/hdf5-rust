//! Create, and continuously write several chunks to a chunked 1D dataset
//!
//! Finally read it, get the chunking metadata, and use it to read slices aligned to the chunk size

#[cfg(feature = "1.10.5")]
fn main() -> hdf5_metno::Result<()> {
    example::write_hdf5()?;
    example::read_hdf5()?;
    Ok(())
}

#[cfg(not(feature = "1.10.5"))]
fn main() {
    println!(
        "needs version 1.10.5 or later for querying the number of chunks with H5Dget_num_chunks"
    );
}

#[cfg(feature = "1.10.5")]
mod example {
    use hdf5::{File, Result};
    use hdf5_metno::{self as hdf5, Extent};
    use ndarray::Array1;

    // Shared between read_hdf5() and write_hdf5(), otherwise they only rely on file metadata
    const FILE_NAME: &str = "continuous_chunks.h5";
    const DATASET_NAME: &str = "count";

    const CHUNK_SIZE: usize = 5;
    const NUM_CHUNKS: usize = 3; // total chunks to write

    pub fn write_hdf5() -> Result<()> {
        println!("Creating file '{FILE_NAME}' with 1D resizable dataset");
        let file = File::create(FILE_NAME)?;
        let shape = Extent::resizable(0); // 1D resizable
        let ds =
            file.new_dataset::<usize>().chunk((CHUNK_SIZE,)).shape(shape).create(DATASET_NAME)?;

        // Simulate continuously accumulating data in a buffer
        // and writing it to the dataset anytime there's enough to fill a chunk
        let mut buf = Vec::with_capacity(CHUNK_SIZE);
        for i in 0..NUM_CHUNKS * CHUNK_SIZE {
            buf.push(i);
            if buf.len() == CHUNK_SIZE {
                let current_size = ds.size();
                println!("[{i}] writing new chunk, current size = {current_size}");
                ds.resize((current_size + CHUNK_SIZE,))?;
                ds.write_slice(&buf, current_size..current_size + CHUNK_SIZE)?;
                buf.clear();
            }
        }

        Ok(())
    }

    pub fn read_hdf5() -> Result<()> {
        println!("Reading file '{FILE_NAME}'");
        let file = File::open(FILE_NAME)?;
        let ds = file.dataset(DATASET_NAME)?;

        // Check shape
        let shape = ds.shape();
        println!("Dataset shape: {:?}", shape);
        assert_eq!(shape, &[CHUNK_SIZE * NUM_CHUNKS]);

        // Get chunking metadata
        let chunk_size = ds.chunk().unwrap()[0];
        println!("Chunk size: {chunk_size}");
        assert_eq!(chunk_size, CHUNK_SIZE);

        let num_chunks = ds.num_chunks().unwrap();
        println!("Number of chunks: {num_chunks}");
        assert_eq!(num_chunks, NUM_CHUNKS);

        // Read the dataset chunk for chunk
        for chunk_idx in 0..num_chunks {
            let chunk_start = chunk_idx * chunk_size;
            let arr: Array1<usize> = ds.read_slice(chunk_start..chunk_start + chunk_size)?;
            println!("Dataset Chunk #{chunk_idx}: {arr:?}");
            assert_eq!(arr, Array1::from_iter(chunk_start..chunk_start + chunk_size));
        }

        // Read the full dataset in one go
        let arr: Array1<usize> = ds.read_1d()?;
        println!("Full dataset: {arr:?}");
        assert_eq!(arr, Array1::from_iter(0..CHUNK_SIZE * NUM_CHUNKS));

        Ok(())
    }
}
