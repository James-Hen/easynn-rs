pub mod shape;
pub use shape::{ Shape, ShapeTrait };

/// Index: the index of a RANK-ranked tensor.
pub type Index<const RANK: usize> = [usize; RANK];

pub use std::ops::{ Add, Mul };

pub mod num;
pub use num::NumT;

/// Tensor: a generic describing a RANK-ranked tensor.
pub struct Tensor<T: NumT, const RANK: usize> {
    pub(crate) flattened: Vec<T>,
    pub(crate) shape: Shape<RANK>,
}

pub mod error;
pub use error::OutOfBondError;
type Result<T> = std::result::Result<T, OutOfBondError>;

impl<T: NumT, const RANK: usize> Tensor<T, RANK> {
    pub(crate) fn index2pos(&self, at: Index<RANK>) -> Result<usize> {
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
    pub(crate) fn pos2index(&self, mut pos: usize) -> Result<Index<RANK>> {
        if pos > self.flattened.len() {
            return Err(OutOfBondError);
        }
        let mut ind: Index<RANK> = [0; RANK];
        for dimention in (0..RANK).rev() {
            ind[dimention] = pos % self.shape[dimention];
            pos /= self.shape[dimention];
        }
        Ok(ind)
    }

    pub fn zeros(shape: Shape<RANK>) -> Tensor<T, RANK> {
        Tensor::<T, RANK> { flattened: vec![T::zero(); shape.size()], shape: shape, }
    }
    pub fn get(&self, at: Index<RANK>) -> Result<T> {
        let pos = self.index2pos(at)?;
        Ok(self.flattened[pos])
    }
    pub fn set(&mut self, at: Index<RANK>, val: T) -> Result<()> {
        let pos = self.index2pos(at)?;
        self.flattened[pos] = val;
        Ok(())
    }
    pub fn get_shape(&self) -> &Shape<RANK> {
        &self.shape
    }
}