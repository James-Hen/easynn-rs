use std::ops::Index;

/// Shape: describes the shape of a tensor given the rank,
/// which is the dimention count of the tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    bound: Vec<usize>,
}

impl Index<usize> for Shape {
    type Output = usize;
    fn index(&self, ind: usize) -> &Self::Output {
        &self.bound[ind]
    }
}

impl Shape {
    /// Create a Shape object described by an array
    pub fn new<const RANK: usize>(s: [usize; RANK]) -> Self {
        Shape { bound: s.to_vec() }
    }

    /// Return the element counts of the tensor in memory,
    /// e.g.:
    /// 
    /// ```rust
    ///     use easynn::tensor::Shape;
    ///     let s = Shape::new([2, 3, 5]);
    ///     assert_eq!(s.size(), 30_usize);
    /// ```
    pub fn size(&self) -> usize {
        let mut ret: usize = 1;
        for b in &self.bound {
            ret *= b;
        }
        ret
    }
}

#[macro_export]
macro_rules! sh {
    ($t: tt) => {
        &Shape::new($t)
    }
}