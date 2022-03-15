//! The models module
//! 

pub mod sequential;

use crate::layers::*;

pub trait Model<T: NumT>  {
    fn predict(&self, input: &Tensor<T>) -> Result<Tensor<T>>;
}