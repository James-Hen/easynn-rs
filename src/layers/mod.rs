//! The layers module

pub mod dense;

pub use crate::tensor::*;
pub use crate::tensor::error::ShapeMismatchError;
pub type Result<T> = std::result::Result<T, ShapeMismatchError>;

pub trait Layer<InputT: NumT, OutputT: NumT, const INPUT_RANK: usize, const OUTPUT_RANK: usize> {
    fn predict(&self, input: &Tensor<InputT, INPUT_RANK>) -> Result<Tensor<OutputT, OUTPUT_RANK>>;
}