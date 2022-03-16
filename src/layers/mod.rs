//! The layers module

pub mod dense;
pub mod activation;
pub use activation::*;

pub use crate::tensor::*;
pub use crate::tensor::error::ShapeMismatchError;
pub type Result<T> = std::result::Result<T, ShapeMismatchError>;

pub trait Layer<T: NumT> {
    /// Forward-propagate takes an input and calculates the output tensor
    /// 
    /// The third param decides if the result is a^l or z^l
    /// z^l is not subject to acivation function, and a^l = sigma(z^l)
    /// 
    /// Doing activation here is generally faster than doing that later
    fn forward_propagate(&self, input: &Tensor<T>, activate: bool) -> Result<Tensor<T>>;

    /// This is used when training: should get both a^l and z^l
    fn activate(&self, output: &Tensor<T>) -> Result<Tensor<T>>;

    /// Backpropagate takes the delta of output and calculates the delta of the input
    /// 
    /// It relys on the output z of the last layer and the activation of the last layer
    fn backpropagate_delta(&self, delta: &Tensor<T>, a_lst: &Tensor<T>, sigma_lst: &Activation<T>) -> Result<Tensor<T>>;

    /// Do the learning of each layer
    fn descend(&mut self, rate: T, delta: &Tensor<T>, a_lst: &Tensor<T>) -> Result<()>;
}