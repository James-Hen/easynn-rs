//! The module that contains loss functions, e.g., MSE, KL, cross-entropy, hinge, etc.
//!

use crate::tensor::*;
type Result<T> = std::result::Result<T, ShapeMismatchError>;

#[derive(Debug, Copy, Clone)]
pub enum Loss {
    MeanSquare,
}

fn mse<T: NumT>(output: &Tensor::<T>, truth: &Tensor::<T>) -> Result<T> {
    if output.shape != truth.shape {
        return Err(ShapeMismatchError);
    }
    let mut ret = T::zero();
    let len = T::from(output.shape.size()).unwrap();
    for (o, t) in output.flattened.iter().zip(truth.flattened.iter()) {
        ret += (*o - *t).powf(T::one()+T::one());
    }
    Ok(ret.sqrt() / len)
}

fn dmse<T: NumT>(output: &Tensor::<T>, truth: &Tensor::<T>) -> Result<Tensor::<T>> {
    if output.shape != truth.shape {
        return Err(ShapeMismatchError);
    }
    let mut ret = Tensor::<T>::zeros(&truth.shape);
    let len = T::from(output.shape.size()).unwrap();
    for (r, (o, t)) in ret.flattened.iter_mut().zip(output.flattened.iter().zip(truth.flattened.iter())) {
        *r = (*o - *t) * *o * (T::one()+T::one()) / len;
    }
    Ok(ret)
}

impl Loss {
    pub fn call<T: NumT>(&self, output: &Tensor::<T>, truth: &Tensor::<T>) -> Result<T> {
        match self {
            Loss::MeanSquare => mse::<T>(output, truth),
            // _ => T::zero(),
        }
    }
    pub fn diff<T: NumT>(&self, output: &Tensor::<T>, truth: &Tensor::<T>) -> Result<Tensor::<T>> {
        match self {
            Loss::MeanSquare => dmse::<T>(output, truth),
            // _ => T::zero(),
        }
    }
}