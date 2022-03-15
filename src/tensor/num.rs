/// NumT trait is implemented for the numeric types that
/// are accepted to be contained in a tensor.

extern crate num_traits;
use num_traits::{ NumOps, NumAssignOps };
use num_traits::identities::{ One, Zero };

use std::marker::{ Send, Sync };
use std::fmt::Debug;

pub trait NumT:
    PartialEq + Zero + One + NumOps + NumAssignOps
    + Copy + Send + Sync + Debug
{ }

macro_rules! int_trait_impl {
    ($name:ident for $($t:ty)*) => ($(
        impl $name for $t { }
    )*)
}

int_trait_impl!(NumT for usize u8 u16 u32 u64 isize i8 i16 i32 i64 f32 f64);
#[cfg(has_i128)]
int_trait_impl!(NumT for u128 i128);