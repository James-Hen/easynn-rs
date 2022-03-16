//! The module that contains loss functions, e.g., MSE, KL, cross-entropy, hinge, etc.
//!

use crate::tensor::*;
type Result<T> = std::result::Result<T, ShapeMismatchError>;

#[derive(Debug, Copy, Clone)]
pub enum Loss {
    MSE,
}

fn mse<T: NumT>(output: &Tensor::<T>, truth: &Tensor::<T>) -> Result<T> {
    if output.shape != truth.shape {
        return Err(ShapeMismatchError);
    }
    Ok(T::zero())
}

fn dmse<T: NumT>(output: &Tensor::<T>, truth: &Tensor::<T>) -> Result<Tensor::<T>> {
    if output.shape != truth.shape {
        return Err(ShapeMismatchError);
    }
    Ok(Tensor::<T>::zeros(&truth.shape))
}

impl Loss {
    pub fn call<T: NumT>(&self, output: &Tensor::<T>, truth: &Tensor::<T>) -> Result<T> {
        match self {
            MSE => mse::<T>(output, truth),
            // _ => T::zero(),
        }
    }
    pub fn diff<T: NumT>(&self, output: &Tensor::<T>, truth: &Tensor::<T>) -> Result<Tensor::<T>> {
        match self {
            MSE => dmse::<T>(output, truth),
            // _ => T::zero(),
        }
    }
}