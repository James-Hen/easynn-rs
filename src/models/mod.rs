//! The models module
//! 

pub mod sequential;

pub mod losses;

use crate::layers::*;
pub use losses::*;

pub trait Model<T: NumT>  {
    /// Calculate the output according to the input
    fn predict(&self, input: &Tensor<T>) -> Result<Tensor<T>>;
    // Train the model using a input-output pair
    // fn train_once(&mut self, input: &Tensor<T>, output: &Tensor<T>);
}