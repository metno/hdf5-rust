use hdf5_metno_derive::H5Type;

use std::marker::PhantomData;

#[derive(H5Type)]
//~^ ERROR proc-macro derive
//~^^ HELP Cannot derive H5Type for empty tuple structs
struct Foo<T: 'static>(PhantomData<T>);

fn main() {}
