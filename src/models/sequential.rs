//! Sequential model is a linear concatenation of layers.

use crate::models::*;

pub struct Sequential<T: NumT> {
    seq: Vec<Box<dyn Layer<T>>>,
}

impl<T: NumT> Sequential<T> {
    pub fn new() -> Self {
        Sequential::<T> { seq: Vec::<Box<dyn Layer<T>>>::new() }
    }
    pub fn add<L: 'static + Layer<T>>(&mut self, layer: L) {
        self.seq.push(Box::new(layer));
    }
}

impl<T: NumT> Model<T> for Sequential<T> {
    fn predict(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        let mut last_output: Box<Tensor<T>>;
        let mut output: Box<Tensor<T>> = Box::new((*input).clone());
        for layer in &self.seq {
            last_output = output;
            output = Box::new(layer.forward_propagate(&last_output, true).unwrap());
        }
        Ok(*output)
    }
}

#[test]
fn test_sequential_predict() {
    let i_shape = Shape::new([2, 3]);
    let hid_shape =Shape::new([3]);
    let o_shape = Shape::new([2]);

    let input = Tensor::<f64>::new(&i_shape, vec![
        1., 7., 8.,
        -2., 3., 5.,
    ]).unwrap();
    let output = Tensor::<f64>::new(&o_shape, vec![70., 70.]).unwrap();

    let mut nn = Sequential::<f64>::new();
    use crate::layers::activation::Activation::*;
    nn.add(crate::layers::dense::Dense::<f64>::new(&i_shape, &hid_shape, No));
    nn.add(crate::layers::dense::Dense::<f64>::new(&hid_shape, &o_shape, Relu));

    assert_eq!(nn.predict(&input).unwrap(), output);
}