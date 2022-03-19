use crate::layers::*;
use crate::layers::activation::*;

extern crate crossbeam;
extern crate num_cpus;
extern crate rayon;

use rayon::prelude::*;
use rand::Rng;

use std::cmp;

/// This is used to determine how much threads to spawn.
/// 
/// The length argument is the count of total works (e.g. mul counts)
///
/// Need to consider SIMD.
fn determine_thread(len: usize) -> usize {
    const FALL_BACK_SIZE: usize = 256;
    let ncpu = num_cpus::get();
    if len / ncpu < FALL_BACK_SIZE {
        return 1;
    }
    cmp::min(ncpu, len / FALL_BACK_SIZE)
}

/// Weight are arranged in flattened style:
/// every i^th consecutive (input size) items are the weight
/// of the input tensor to the i^th one in the output tensor,
/// e.g.:
///
///  - input: `[01,02;11,12]`
///  - output: `[21,22;31,32]`
///  - weight: `[21~01,21~02,21~11,21~12;22~01,22~02,22~11,22~12;...;...]`
///  - bias: `[21,22;31;32]`
///
/// When processed in parallel, each output coordinates a `mult`,
/// and each `chunk` includes many `mult`.
/// 
#[derive(Debug)]
pub struct Dense<T: NumT> {
    pub(crate) input_shape: Shape,
    pub(crate) output_shape: Shape,
    pub(crate) weight: Vec<T>,
    pub(crate) bias: Vec<T>,
    pub(crate) activation: Activation<T>,
}

impl<T: NumT> Dense<T> {
    pub fn new(i_shape: &Shape, o_shape: &Shape, act: Activation<T>) -> Self {
        let ilen = i_shape.size();
        let olen = o_shape.size();
        let mut rng = rand::thread_rng();
        Dense::<T> {
            input_shape: i_shape.clone(),
            output_shape: o_shape.clone(),
            weight: (0..ilen*olen).map(|_| T::from(rng.gen_range(0.0..=1.0)).unwrap()).collect(),
            bias: (0..olen).map(|_| T::from(rng.gen_range(0.0..=1.0)).unwrap()).collect(),
            activation: act,
        }
    }
}

/// Helpers like `slice_iter(w, len, j)` is implemented to access the weight slice j,
/// containing len(== input length) elements
macro_rules! slice_iter {
    ($w: expr, $len: expr, $j: expr) => {
        $w[$j*$len..($j+1)*$len].into_iter()
    }
}
macro_rules! slice_iter_mut {
    ($w: expr, $len: expr, $j: expr) => {
        $w[$j*$len..($j+1)*$len].iter_mut()
    }
}

impl<T: NumT> Layer<T> for Dense<T> {
    fn get_activation(&self) -> Activation<T> {
        self.activation
    }
    fn get_input_shape(&self) -> Shape {
        self.input_shape.clone()
    }
    fn get_output_shape(&self) -> Shape {
        self.output_shape.clone()
    }
    fn get_weight_count(&self) -> usize {
        self.weight.len()
    }

    fn forward_propagate(&self, input: &Tensor<T>, activate: bool) -> Result<Tensor<T>> {
        if input.shape != self.input_shape {
            return Err(ShapeMismatchError);
        }
        let mut output = Tensor::<T>::zeros(&self.output_shape);
        let olen = output.flattened.len();
        let ilen = input.flattened.len();

        let threads = determine_thread(ilen * olen);
        let mults_per_chunk = olen / threads + 1;
        {
            let o_chunks = output.flattened.chunks_mut(mults_per_chunk);
            let w_chunks = self.weight.chunks(mults_per_chunk * ilen);
            crossbeam::scope(|spawner| {
                for (i, (o_chk, w_chk)) in o_chunks.zip(w_chunks).enumerate() {
                    spawner.spawn(move |_| {
                        for (j, o) in o_chk.into_iter().enumerate() {
                            *o = self.bias[i*mults_per_chunk + j];
                            // Do o = input.dot(w_chk[j])
                            for (k, &w) in slice_iter!(w_chk, ilen, j).enumerate() {
                                *o += w * input.flattened[k];
                            }
                            if activate {
                                *o = self.activation.call(*o);
                            }
                        }
                    });
                }
            }).unwrap(); 
        }
        Ok(output)
    }
    fn activate(&self, output: &Tensor<T>) -> Result<Tensor<T>> {
        if output.shape != self.output_shape {
            return Err(ShapeMismatchError);
        }
        let mut act_vec = vec![T::zero(); output.shape.size()];
        act_vec.par_iter_mut().zip(output.flattened.par_iter()).for_each(|(a, o)| {
            *a = self.activation.call(*o);
        });
        Ok(Tensor::<T>::new(&self.output_shape, act_vec))
    }
    fn backpropagate_delta(&self, delta: &Tensor<T>, z_lst: &Tensor<T>, sigma_lst: &Activation<T>) -> Result<Tensor<T>> {
        if delta.shape != self.output_shape || z_lst.shape != self.input_shape {
            return Err(ShapeMismatchError);
        }
        let ilen = self.input_shape.size();
        let dlen = delta.flattened.len();

        // calculate products of weight and delta, to be sumed
        let mut prod = vec![T::zero(); self.weight.len()];
        let threads = determine_thread(ilen * dlen);
        let mults_per_chunk = dlen / threads + 1;
        {
            let d_chunks = delta.flattened.chunks(mults_per_chunk); // delta chunk
            let w_chunks = self.weight.chunks(mults_per_chunk * ilen); // weight chunk
            let p_chunks = prod.chunks_mut(mults_per_chunk * ilen); // prod chunk
            crossbeam::scope(|spawner| {
                for ((w_chk, p_chk), d_chk) in w_chunks.zip(p_chunks).zip(d_chunks) {
                    spawner.spawn(move |_| {
                        for (j, &d) in d_chk.into_iter().enumerate() {
                            // p[j] = w[j] * delta 
                            for (&w, p) in slice_iter!(w_chk, ilen, j).zip(slice_iter_mut!(p_chk, ilen, j)) {
                                *p = w * d;
                            }
                        }
                    });
                }
            }).unwrap(); 
        }

        // add those slices back
        let sum_prod = prod.par_chunks_mut(ilen).reduce_with(
            |s1, s2| {
                let len = s1.len();
                for i in 0..len {
                    let s = s1[i] + s2[i];
                    s1[i] = s; s2[i] = s;
                }
                s1
            }
        ).unwrap();

        let mut lst_delta = Tensor::<T>::zeros(&self.input_shape);
        lst_delta.flattened = sum_prod.to_vec();
        
        // dot product sigma-1(z^l) and w^Td^{l+1}
        lst_delta.flattened.par_iter_mut().zip(z_lst.flattened.par_iter()).for_each(|(d, z)| {
            *d *= sigma_lst.diff(*z);
        });

        Ok(lst_delta)
    }
    fn add_weight_delta_to(&self, delta: &Tensor<T>, a_lst: &Tensor<T>, cum_dw: &mut Vec<T>, cum_db: &mut Tensor<T>) -> Result<()> {
        if cum_dw.len() != delta.shape.size() * a_lst.shape.size() || cum_db.shape != delta.shape {
            return Err(ShapeMismatchError);
        }
        // compute d dot a^T
        let dlen = delta.flattened.len();
        let alen = a_lst.flattened.len();
        let threads = determine_thread(dlen * alen);
        let d_per_chunk = dlen / threads + 1;
        {
            let d_chunks = delta.flattened.chunks(d_per_chunk);
            let w_chunks = cum_dw.chunks_mut(d_per_chunk * alen);
            crossbeam::scope(|spawner| {
                for (d_chk, w_chk) in d_chunks.zip(w_chunks) {
                    spawner.spawn(move |_| {
                        for (j, d) in d_chk.into_iter().enumerate() {
                            // Do w_chk[j] += a * d[j]
                            for (k, w) in slice_iter_mut!(w_chk, alen, j).enumerate() {
                                *w += *d * a_lst.flattened[k];
                            }
                        }
                    });
                }
            }).unwrap(); 
        }
        // add delta to cum_db
        cum_db.flattened.par_iter_mut().zip(delta.flattened.par_iter()).for_each(|(db, d)| {
            *db += *d;
        });
        Ok(())
    }
    fn descend(&mut self, rate: T, dw: &Vec<T>, db: &Tensor<T>) -> Result<()> {
        if db.shape != self.output_shape || dw.len() != self.weight.len() {
            return Err(ShapeMismatchError);
        }
        // do weight update
        self.weight.par_iter_mut().zip(dw.par_iter()).for_each(|(wi, dwi)| {
            *wi -= rate * *dwi;
        });

        // do bias update
        self.bias.par_iter_mut().zip(db.flattened.par_iter()).for_each(|(bi, dbi)| {
            *bi -= rate * *dbi;
        });

        Ok(())
    }
}

#[test]
fn test_dense_forward() {
    let input = Tensor::<f64>::new(&Shape::new([2, 3]), vec![
        1., 7., 8.,
        -2., 3., 5.,
    ]);
    let l = Dense::<f64> {
        input_shape: Shape::new([2, 3]),
        output_shape: Shape::new([2]),
        weight: vec![
            2., 1., -1., 3., 2., 1.,
            1., 0., 0., -2., 1., 0.,
        ],
        bias: vec![-5., -1.],
        activation: Activation::<f64>::No,
    };
    let output = Tensor::<f64>::new(&Shape::new([2]), vec![1., 7.]);
    assert_eq!(l.forward_propagate(&input, true).unwrap(), output);
}

#[test]
fn test_dense_activate() {
    let l = Dense::<f64> {
        input_shape: Shape::new([2, 3]),
        output_shape: Shape::new([3, 4]),
        weight: vec![0.; 12_usize],
        bias: vec![0.; 12_usize],
        activation: Activation::<f64>::Sigmoid,
    };
    let output = Tensor::<f64>::new(&Shape::new([3, 4]), vec![
        -3., -2., -1., 0.,
        1., 2., 3., 4.,
        5., 6., 7., 8.,
    ]);
    let mut ans_vec = vec![0.; 12];
    for (y, x) in ans_vec.iter_mut().zip(output.flattened.iter()) {
        *y = Activation::<f64>::Sigmoid.call(*x);
    }
    let answer = Tensor::<f64>::new(&Shape::new([3, 4]), ans_vec);
    assert_eq!(l.activate(&output).unwrap(), answer);
}

#[test]
fn test_dense_backpropagate() {
    let lst_a = Tensor::<f64>::new(&Shape::new([2, 3]), vec![
        1., 7., 8.,
        -2., 3., 5.,
    ]);
    let l = Dense::<f64> {
        input_shape: Shape::new([2, 3]),
        output_shape: Shape::new([2]),
        weight: vec![
            2., 1., -1., 3., 2., 1.,
            1., 0., 0., -2., 1., 0.,
        ],
        bias: vec![-5., -1.],
        activation: Activation::<f64>::No,
    };
    let delta = Tensor::<f64>::new(&Shape::new([2]), vec![1., 7.]);
    let answer = Tensor::<f64>::new(&Shape::new([2, 3]), vec![
        9., 1., -1.,
        0., 9., 1.,
    ]);
    assert_eq!(l.backpropagate_delta(&delta, &lst_a, &Activation::<f64>::Relu).unwrap(), answer);
}

#[test]
fn test_dense_descend() {
    let da_lst = vec![
        1., 7., 8.,
        -2., 3., 5.,
        
        7., 49., 56.,
        -14., 21., 35.,
    ];
    let mut l = Dense::<f64> {
        input_shape: Shape::new([2, 3]),
        output_shape: Shape::new([2]),
        weight: vec![
            2., 1., -1., 3., 2., 1.,
            1., 0., 0., -2., 1., 0.,
        ],
        bias: vec![-5., -1.],
        activation: Activation::<f64>::No,
    };
    let delta = Tensor::<f64>::new(&Shape::new([2]), vec![1., 7.]);
    l.descend(0.1, &da_lst, &delta).unwrap();
    let w_ans = vec![
        2.-0.1, 1.-0.7, -1.-0.8, 3.+0.2, 2.-0.3, 1.-0.5,
        1.-0.7, 0.-4.9, 0.-5.6, -2.+1.4, 1.-2.1, 0.-3.5,
    ];
    let b_ans = vec![-5.-0.1, -1.-0.7];
    let eps = 1e-8;
    for (w, upd) in w_ans.into_iter().zip(l.weight.into_iter()) {
        assert!(
            (w - upd).abs() < eps,
            "expected {}, got {}", w, upd
        );
    }
    for (b, upd) in b_ans.into_iter().zip(l.bias.into_iter()) {
        assert!(
            (b - upd).abs() < eps,
            "expected {}, got {}", b, upd
        );
    }
}

#[test]
fn test_add_weight_delta_to() {
    let a_lst = Tensor::<f64>::new(&Shape::new([2, 3]), vec![
        1., 7., 8.,
        -2., 3., 5.,
    ]);
    let l = Dense::<f64> {
        input_shape: Shape::new([2, 3]),
        output_shape: Shape::new([2]),
        weight: vec![
            2., 1., -1., 3., 2., 1.,
            1., 0., 0., -2., 1., 0.,
        ],
        bias: vec![-5., -1.],
        activation: Activation::<f64>::No,
    };
    let delta = Tensor::<f64>::new(&Shape::new([2]), vec![1., 7.]);
    let mut cum_dw = vec![0.; 12];
    let mut cum_db = Tensor::<f64>::new(&Shape::new([2]), vec![0., 0.]);
    l.add_weight_delta_to(&delta, &a_lst, &mut cum_dw, &mut cum_db).unwrap();
    let ans_da_lst = vec![
        1., 7., 8.,
        -2., 3., 5.,
        
        7., 49., 56.,
        -14., 21., 35.,
    ];
    assert_eq!(cum_dw, ans_da_lst);
    assert_eq!(cum_db, delta);
}