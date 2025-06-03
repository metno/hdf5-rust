#[cfg(feature = "1.10.2")]
use hdf5_metno as hdf5;

#[cfg(feature = "1.10.2")]
fn reader() {
    let file = hdf5::File::open_as("swmr.h5", hdf5::OpenMode::ReadSWMR).unwrap();
    println!("Reader: Opened file");
    let var = file.dataset("foo").unwrap();

    for _ in 0..5 {
        var.refresh().unwrap();
        let shape = var.shape();
        println!("Reader: Got shape: {shape:?}");
        // If reading one should use the shape directly without
        // using the convenience read_2d etc. functions which
        // might get confused if the shape is changed during reading
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
}

fn main() {
    #[cfg(not(feature = "1.10.2"))]
    println!("This examples requires hdf5 >= 1.10.2 to enable SWMR and set libver_bounds");

    #[cfg(feature = "1.10.2")]
    {
        let file = hdf5::File::with_options()
            .with_fapl(|fapl| fapl.libver_v110())
            .create("swmr.h5")
            .unwrap();

        let var = file.new_dataset::<u8>().shape((0.., 5)).create("foo").unwrap();

        file.start_swmr().unwrap();
        println!("Writer: Wrote file");

        let thread = std::thread::spawn(|| reader());

        for i in 0..5 {
            var.resize((i + 1, 5)).unwrap();
            var.write_slice(&[i, i, i, i, i], (i, 0..)).unwrap();
            var.flush().unwrap();
            println!("Writer: Wrote {i}");
            std::thread::sleep(std::time::Duration::from_secs(5));
        }

        thread.join().unwrap();
    }
}
