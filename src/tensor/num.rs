/// NumT trait is implemented for the numeric types that
/// are accepted to be contained in a tensor.

extern crate num_traits;
use num_traits::{ NumOps, NumAssignOps, Float };
use num_traits::identities::{ One, Zero };

use std::ops::{ Neg };
use std::marker::{ Send, Sync };
use std::fmt::{ Debug, Display };

pub trait NumT:
    PartialEq + PartialOrd + Zero + One + NumOps + NumAssignOps
    + Copy + Send + Sync + Debug + Display + Neg + Float
{ }

macro_rules! trait_impl {
    ($name:ident for $($t:ty)*) => ($(
        impl $name for $t { }
    )*)
}

trait_impl!(NumT for f32 f64);