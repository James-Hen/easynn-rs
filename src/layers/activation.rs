//! The module that contains activation functions, e.g. sigmoid, relu, etc.
//!

use crate::tensor::num::*;

#[derive(Debug, Copy, Clone)]
pub enum Activation<T: NumT> {
    No,
    Sigmoid,
    Tanh,
    Relu,
    LeakyRelu(T),
}

fn sigmoid<T: NumT>(x: T) -> T {
    T::one() / (T::one() + (-x).exp())
}

fn dsigmoid<T: NumT>(x: T) -> T {
    sigmoid(x) * (T::one() - sigmoid(x))
}

fn tanh<T: NumT>(x: T) -> T {
    let ex = x.exp();
    let nex = (-x).exp();
    (ex - nex) / (ex + nex)
}

fn dtanh<T: NumT>(x: T) -> T {
    let thx = tanh(x);
    T::one() - thx * thx
}

fn relu<T: NumT>(x: T) -> T {
    if x < T::zero() { T::zero() } else { x }
}

fn drelu<T: NumT>(x: T) -> T {
    if x < T::zero() { T::zero() } else { T::one() }
}

fn leaky_relu<T: NumT>(a: T, x: T) -> T {
    if x < T::zero() { T::zero() } else { a * T::one() }
}

fn dleaky_relu<T: NumT>(a: T, x: T) -> T {
    if x < T::zero() { T::zero() } else { a }
}

use Activation::*;

impl<T: NumT> Activation<T> {
    pub fn call(&self, x: T) -> T {
        match self {
            No => x,
            Sigmoid => sigmoid::<T>(x),
            Tanh => tanh::<T>(x),
            Relu => relu::<T>(x),
            LeakyRelu(a) => leaky_relu::<T>(*a, x),
            // _ => T::zero(),
        }
    }
    pub fn diff(&self, x: T) -> T {
        match self {
            No => T::one(),
            Sigmoid => dsigmoid::<T>(x),
            Tanh => dtanh::<T>(x),
            Relu => drelu::<T>(x),
            LeakyRelu(a) => dleaky_relu::<T>(*a, x),
            // _ => T::zero(),
        }
    }
}