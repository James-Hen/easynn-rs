//! Sequential model is a linear concatenation of layers.

use crate::models::*;
extern crate itertools;
extern crate rayon;
use rayon::prelude::*;
use itertools::Itertools;

use crate::layers::*;

pub struct Sequential<T: NumT> {
    seq: Vec<Box<dyn Layer<T>>>,
    loss: Loss,
}

impl<T: NumT> Sequential<T> {
    pub fn new(l: Loss) -> Self {
        Sequential::<T> {
            seq: Vec::<Box<dyn Layer<T>>>::new(),
            loss: l,
        }
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
    fn propagate_sample(&self, input: &Tensor<T>, truth: &Tensor<T>) -> Result<(Vec<Tensor<T>>, Vec<Tensor<T>>)> {
        // forward propagate
        let mut a_lst = Vec::<Tensor<T>>::new();
        let mut z_l = Vec::<Tensor<T>>::new();
        a_lst.push((*input).clone());
        for layer in &self.seq {
            let z_now = layer.forward_propagate(&a_lst.last().unwrap(), false)?;
            z_l.push(z_now);
            a_lst.push(layer.activate(&z_l.last().unwrap())?);
        }
        let mut d_lrev = Vec::<Tensor<T>>::new();
        d_lrev.push(self.loss.diff(&a_lst.last().unwrap(), truth)?);
        let mut z_lst_iter = z_l.iter().rev();
        z_lst_iter.next().unwrap();
        // backward propagate
        for ((layer, layer_lst), zlst) in self.seq.iter().rev().tuple_windows().zip(z_lst_iter) {
            d_lrev.push(
                layer.backpropagate_delta(
                    &d_lrev.last().unwrap(), zlst, &layer_lst.get_activation()
                ).unwrap()
            );
        }
        d_lrev.reverse();
        Ok((d_lrev, a_lst))
    }
    fn update_delta_da(&self, cum_dw: &mut Vec<Vec<T>>, cum_db: &mut Vec<Tensor<T>>, delta: &Vec<Tensor<T>>, a_lst: &Vec<Tensor<T>>) {
        for (layer, ((d, alst), (cumdw, cumdb))) in self.seq.iter().zip(
            delta.iter().zip(a_lst.iter()).zip(
                cum_dw.iter_mut().zip(cum_db.iter_mut())
            )
        ) {
            layer.add_weight_delta_to(d, alst, cumdw, cumdb).unwrap();
        }
    }
    fn descend(&mut self, rate: T, dw: &Vec<Vec<T>>, db: &Vec<Tensor<T>>) {
        for (layer, (dwi, dbi)) in self.seq.iter_mut().zip(dw.iter().zip(db.iter())) {
            layer.descend(rate, dwi, dbi).unwrap();
        }
    }
    fn train_once(&mut self, inputs: &Vec<Tensor<T>>, truths: &Vec<Tensor<T>>, batch_size: usize, learning_rate: T, verbose: bool) {
        // prepare the intermediate accumulators
        let mut cum_dw = Vec::<Vec<T>>::new();
        let mut cum_db = Vec::<Tensor<T>>::new();
        for layer in &self.seq {
            cum_dw.push(vec![T::zero(); layer.get_weight_count()]);
            cum_db.push(Tensor::<T>::zeros(&layer.get_output_shape()));
        }

        // assert_eq!(inputs.len(), truths.len());
        let in_batches = inputs.chunks(batch_size);
        let tr_batches = truths.chunks(batch_size);
        for (i, (in_batch, tr_batch)) in in_batches.into_iter().zip(tr_batches.into_iter()).enumerate() {
            let mut tot_loss = T::zero();
            if verbose {
                print!("Trainning batch {} ... ", i);
            }
            // batch size
            let bsize = T::from(in_batch.len()).unwrap();
            // clear the cumulators
            for cum_dw_l in &mut cum_dw {
                cum_dw_l.par_iter_mut().for_each(|cdw| { *cdw = T::zero(); });
            }
            for cum_db_l in &mut cum_db {
                cum_db_l.flattened.par_iter_mut().for_each(|cdb| { *cdb = T::zero(); });
            }
            // train for a batch
            for (input, truth) in in_batch.into_iter().zip(tr_batch.into_iter()) {
                let (deltas, mut interoutputs) = self.propagate_sample(input, truth).unwrap();
                let result = interoutputs.pop().unwrap();
                tot_loss += self.loss.call(&result, truth).unwrap();
                self.update_delta_da(&mut cum_dw, &mut cum_db, &deltas, &interoutputs);
            }
            // process the cumulators
            for cum_dw_l in &mut cum_dw {
                cum_dw_l.par_iter_mut().for_each(|cdw| { *cdw *= T::one() / bsize; });
            }
            for cum_db_l in &mut cum_db {
                cum_db_l.flattened.par_iter_mut().for_each(|cdb| { *cdb *= T::one() / bsize; });
            }
            // descend
            self.descend(learning_rate, &cum_dw, &cum_db);

            if verbose {
                println!("Ok, Mean loss ({:?}): {}", self.loss, tot_loss / bsize);
            }
        }
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
    ]);
    let output = Tensor::<f64>::new(&o_shape, vec![70., 70.]);

    let mut nn = Sequential::<f64>::new(Loss::MeanSquare);
    use crate::layers::activation::Activation::*;
    nn.add(crate::layers::dense::Dense::<f64>::new(&i_shape, &hid_shape, No));
    nn.add(crate::layers::dense::Dense::<f64>::new(&hid_shape, &o_shape, Relu));

    assert_eq!(nn.predict(&input).unwrap(), output);
}

/// This test is to test if it can learn the 1 bit xor function
#[test]
fn test_sequential_xor1() {
    use crate::prelude::*;
    use rand::Rng;

    // create the network and add 2 layers
    let mut nn = Sequential::<f64>::new(Loss::MeanSquare);
    // add the input (2 integers) and a hidden layer
    nn.add(Dense::<f64>::new(sh!([2]), sh!([2]), Activation::Sigmoid));
    // add the output layer
    nn.add(Dense::<f64>::new(sh!([2]), sh!([1]), Activation::Sigmoid));

    // create the training set
    let mut rng = rand::thread_rng();
    let tot_samples: usize = 100;
    let mut inputs = vec![Tensor::new(sh!([2]), vec![0_f64, 0_f64]); tot_samples];
    let mut outputs = vec![Tensor::new(sh!([1]), vec![0_f64]); tot_samples];
    for (i, o) in inputs.iter_mut().zip(outputs.iter_mut()) {
        let a = rng.gen::<u32>();
        let b = rng.gen::<u32>();
        i.set([0], (a & 1) as f64);
        i.set([1], (b & 1) as f64);
        o.set([0], ((a ^ b) & 1).into());
    }

    // train the model
    for i in 0..50 {
        println!("[Epoch {}]", i);
        nn.train_once(&inputs, &outputs, 10, 0.1, true);
    }

    // evaluate the model
    let test_in1 = [0., 1., 0., 1.];
    let test_in2 = [0., 0., 1., 1.];
    let test_out = [0., 1., 1., 0.];
    for ((i1, i2), o) in test_in1.iter().zip(test_in2.iter()).zip(test_out.iter()) {
        let test_res = nn.predict(&Tensor::new(sh!([2]), vec![*i1, *i2])).unwrap().get([0]);
        assert_eq!(test_res.round(), *o);
    }
}