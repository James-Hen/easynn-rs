use crate::layers::*;

extern crate crossbeam;
extern crate num_cpus;

/// Weight are arranged in flattened style:
/// every i^th consecutive (size) items are the weight
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
#[derive(Debug)]
pub struct Dense<T: NumT, const INPUT_RANK: usize, const OUTPUT_RANK: usize> {
    input_shape: Shape<INPUT_RANK>,
    output_shape: Shape<OUTPUT_RANK>,
    weight: Vec<T>,
    bias: Vec<T>,
}

impl<T: NumT, const INPUT_RANK: usize, const OUTPUT_RANK: usize>
    Layer<T, T, INPUT_RANK, OUTPUT_RANK> for Dense<T, INPUT_RANK, OUTPUT_RANK>
{
    fn predict(&self, input: &Tensor<T, INPUT_RANK>) -> Result<Tensor<T, OUTPUT_RANK>> {
        if input.shape != self.input_shape {
            return Err(ShapeMismatchError);
        }
        let mut output = Tensor::<T, OUTPUT_RANK>::zeros(&self.output_shape);
        let olen = output.flattened.len();
        let ilen = input.flattened.len();

        let threads = num_cpus::get();
        let mults_per_chunk = olen / threads + 1;
        {
            let o_chunks = output.flattened.chunks_mut(mults_per_chunk);
            let w_chunks = self.weight.chunks(mults_per_chunk * ilen);
            crossbeam::scope(|spawner| {
                for (i, (o_chk, w_chk)) in o_chunks.zip(w_chunks).enumerate() {
                    spawner.spawn(move |_| {
                        for (j, o) in o_chk.into_iter().enumerate() {
                            *o = self.bias[i*mults_per_chunk + j];
                            for (k, &w) in w_chk[j*ilen..(j+1)*ilen].into_iter().enumerate() {
                                *o += w * input.flattened[k];
                            }
                        }
                    });
                }
            }).unwrap(); 
        }
        Ok(output)
    }
}

#[test]
fn test_dense_predict() {
    let input = Tensor::<isize, 2>::new(&[2, 3], vec![
        1, 7, 8,
        -2, 3, 5,
    ]).unwrap();
    let l = Dense::<isize, 2, 1> {
        input_shape: [2, 3],
        output_shape: [2],
        weight: vec![
            2, 1, -1, 3, 2, 1,
            1, 0, 0, -2, 1, 0,
        ],
        bias: vec![-5, -1],
    };
    let output = Tensor::<isize, 1>::new(&[2], vec![1, 7]).unwrap();
    assert_eq!(l.predict(&input).unwrap(), output);
}