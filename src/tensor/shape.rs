/// Shape: describes the shape of a tensor given the rank RANK,
/// which is the dimention count of the tensor.
pub type Shape<const RANK: usize> = [usize; RANK];

/// ShapeTrait: gives an easy way to get the total memory size
/// of a shape.
pub trait ShapeTrait<const RANK: usize> {
    fn size(&self) -> usize;
}
impl<const RANK: usize> ShapeTrait<RANK> for Shape<RANK> {
    fn size(&self) -> usize {
        let mut ret: usize = 1;
        for bound in self {
            ret *= bound;
        }
        ret
    }
}