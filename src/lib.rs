//! This crate aims to provide neuro network developing and training utilities
//! in rust, where a variety of models and layers are supported.
//! 
//! Note: this crate is currently in pre-alpha, any interface is subject to change.
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