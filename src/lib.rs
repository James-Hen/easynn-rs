//! This crate aims to provide neuro network developing and training utilities
//! in rust, where a variety of models and layers are supported.
//! 
//! Note: this crate is currently in pre-alpha, any interface is subject to change.
//! 
//! ## Xor Example
//! 
//! Here we examine an example of building up a 3-layer MLP, learning the xor function of 64 bit integer.
//! 
//! ```rust
//!     use easynn::prelude::*;
//!     use rand::Rng;
//!     
//!     // the target xor function
//!     let target_func = |x: u32, y: u32| { x ^ y };
//!     
//!     // create the network and add 4 layers
//!     let mut nn = Sequential::<f64>::new(Loss::MeanSquare);
//!     // add the input (2 integers) and a hidden layer
//!     nn.add(Dense::<f64>::new(sh!([2]), sh!([2, 32]), Activation::Relu));
//!     // add another hidden layer
//!     nn.add(Dense::<f64>::new(sh!([2, 32]), sh!([1000]), Activation::Relu));
//!     // add another hidden layer
//!     nn.add(Dense::<f64>::new(sh!([1000]), sh!([32]), Activation::Relu));
//!     // add the output layer
//!     nn.add(Dense::<f64>::new(sh!([32]), sh!([1]), Activation::Relu));
//!     
//!     // create the training set
//!     let mut rng = rand::thread_rng();
//!     let tot_samples: usize = 1000;
//!     let mut inputs = vec![Tensor::new(sh!([2]), vec![0_f64, 0_f64]); tot_samples];
//!     let mut outputs = vec![Tensor::new(sh!([1]), vec![0_f64]); tot_samples];
//!     for (i, o) in inputs.iter_mut().zip(outputs.iter_mut()) {
//!         let a = rng.gen::<u32>();
//!         let b = rng.gen::<u32>();
//!         i.set([0], a as f64);
//!         i.set([1], b as f64);
//!         o.set([0], target_func(a, b).into());
//!     }
//! 
//!     // train the model
//!     for _i in 0..10 {
//!         nn.train_once(&inputs, &outputs, 100, 0.1, true);
//!     }
//! 
//!     // evaluate the model
//!     let test_in1: u32 = 19260817;
//!     let test_in2: u32 = 1145141919;
//!     let test_out: u32 = target_func(test_in1, test_in2);
//!     let test_res = nn.predict(&Tensor::new(sh!([2]), vec![test_in1 as f64, test_in2 as f64])).unwrap().get([0]);
//!     dbg!("The prediction of input\n\t{:b} and\n\t{:b} is\n\t{:b} , expected\n\t{:b}"
//!         , test_in1, test_in2, test_res.floor() as u32, test_out);
//! ```
//! 
//! ## Supported models
//!  - [ ] `Sequential`: similar to [The Sequential model](https://www.tensorflow.org/guide/keras/sequential_model) of [Keras](https://keras.io/)
//!
//! ## Supported layer types
//!  - Primitive types:
//!    - [x] `Dense`: fully connected layers
//!  - CNN types:
//!    - [ ] `Conv`: the convolution layer
//!    - [ ] `Pooling`: the pooling layer


pub mod layers;
pub mod models;
pub mod tensor;

pub mod prelude {
    pub use crate::{ sh };
    pub use crate::layers::{ dense::Dense, activation::Activation };
    pub use crate::models::{ Model, sequential::Sequential, losses::Loss };
    pub use crate::tensor::{ shape::{ Shape }, Tensor };
}