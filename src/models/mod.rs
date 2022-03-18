//! The models module
//! 

pub mod sequential;

pub mod losses;

use crate::layers::*;
pub use losses::*;

pub trait Model<T: NumT>  {
    /// Calculate the output according to the input
    fn predict(&self, input: &Tensor<T>) -> Result<Tensor<T>>;
    /// Forward propagate and backward propagate using a input-output pair
    /// 
    /// Returns (the delta of each layer and the output for each layer)
    fn propagate_sample(&self, input: &Tensor<T>, truth: &Tensor<T>) -> Result<(Vec<Tensor<T>>, Vec<Tensor<T>>)>;
    /// Add back the delta of each layer to cum_delta
    /// and add back (d dot a^T) of each layer to cum_da_lst
    fn update_delta_da(&self, cum_dw: &mut Vec<Vec<T>>, cum_db: &mut Vec<Tensor<T>>, delta: &Vec<Tensor<T>>, a_lst: &Vec<Tensor<T>>);
    /// Descend
    fn descend(&mut self, rate: T, dw: &Vec<Vec<T>>, db: &Vec<Tensor<T>>);
    
    /// Trains the model given the dataset by an epoch
    fn train_once(&mut self, inputs: &Vec<Tensor<T>>, truths: &Vec<Tensor<T>>, batch_size: usize, learning_rate: T, verbose: bool);
}