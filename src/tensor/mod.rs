//! The tensor module
//!

pub mod shape;
pub use shape::*;

pub use std::ops::{ Add, Mul };

pub mod num;
pub use num::NumT;

/// Tensor: a generic describing a tensor with the element type T.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T: NumT> {
    pub(crate) shape: Shape,
    pub(crate) flattened: Vec<T>,
}

/// TensorIndex: the index of a RANK-ranked tensor.
pub type TensorIndex<const RANK: usize> = [usize; RANK];

pub mod error;
pub use error::{ OutOfBondError, ShapeMismatchError };
type Result<T> = std::result::Result<T, OutOfBondError>;

impl<T: NumT> Tensor<T> {
    pub(crate) fn index2pos<const RANK: usize>(&self, at: TensorIndex<RANK>) -> Result<usize> {
        let mut pos: usize = 0;
        for dimention in 0..RANK {
            if at[dimention] >= self.shape[dimention] {
                return Err(OutOfBondError);
            }
            pos *= self.shape[dimention];
            pos += at[dimention];
        }
        Ok(pos)
    }
    pub(crate) fn pos2index<const RANK: usize>(&self, mut pos: usize) -> Result<TensorIndex<RANK>> {
        if pos > self.flattened.len() {
            return Err(OutOfBondError);
        }
        let mut ind: TensorIndex<RANK> = [0; RANK];
        for dimention in (0..RANK).rev() {
            ind[dimention] = pos % self.shape[dimention];
            pos /= self.shape[dimention];
        }
        Ok(ind)
    }

    pub fn new(shape: &Shape, flattened: Vec<T>) -> std::result::Result<Self, ShapeMismatchError> {
        if flattened.len() != shape.size() {
            return Err(ShapeMismatchError)
        }
        Ok(Tensor {
            flattened: flattened,
            shape: shape.clone(),
        })
    }
    pub fn zeros(shape: &Shape) -> Self {
        Tensor::<T> { flattened: vec![T::zero(); shape.size()], shape: shape.clone(), }
    }
    pub fn ones(shape: &Shape) -> Self {
        Tensor::<T> { flattened: vec![T::one(); shape.size()], shape: shape.clone(), }
    }
    pub fn get<const RANK: usize>(&self, at: TensorIndex<RANK>) -> Result<T> {
        let pos = self.index2pos(at)?;
        Ok(self.flattened[pos])
    }
    pub fn set<const RANK: usize>(&mut self, at: TensorIndex<RANK>, val: T) -> Result<()> {
        let pos = self.index2pos(at)?;
        self.flattened[pos] = val;
        Ok(())
    }
    pub fn get_shape(&self) -> &Shape {
        &self.shape
    }
}