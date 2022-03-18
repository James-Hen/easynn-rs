//! The layers module

pub mod dense;
pub mod activation;
pub use activation::*;

pub use crate::tensor::*;
pub use crate::tensor::error::ShapeMismatchError;
pub type Result<T> = std::result::Result<T, ShapeMismatchError>;

pub trait Layer<T: NumT> {
    /// Getter method to specify a field `activation`
    fn get_activation(&self) -> Activation<T>;
    /// Getter method to specify a field `input_shape`
    fn get_input_shape(&self) -> Shape;
    /// Getter method to specify a field `output_shape`
    fn get_output_shape(&self) -> Shape;
    /// Get the weight counts
    fn get_weight_count(&self) -> usize;

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
    fn backpropagate_delta(&self, delta: &Tensor<T>, z_lst: &Tensor<T>, sigma_lst: &Activation<T>) -> Result<Tensor<T>>;

    /// Calculates the delta of the weights given the layer's delta and the input,
    /// and add it to the given vector
    fn add_weight_delta_to(&self, delta: &Tensor<T>, a_lst: &Tensor<T>, cum_dw: &mut Vec<T>, cum_db: &mut Tensor<T>) -> Result<()>;

    /// Do the learning of each layer
    fn descend(&mut self, rate: T, dw: &Vec<T>, db: &Tensor<T>) -> Result<()>;
}