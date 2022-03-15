/// Errors that may occur while manipulating tensors

use std::fmt;

#[derive(Debug, Clone)]
pub struct OutOfBondError;

impl fmt::Display for OutOfBondError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Tensor index or position out of bond!")
    }
}

#[derive(Debug, Clone)]
pub struct ShapeMismatchError;

impl fmt::Display for ShapeMismatchError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Tensor shape mismatch!")
    }
}