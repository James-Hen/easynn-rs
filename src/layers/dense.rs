use crate::layers::*;

pub struct Dense<T: NumT, const INPUT_RANK: usize, const OUTPUT_RANK: usize> {
    input_shape: Shape<INPUT_RANK>,
    output_shape: Shape<OUTPUT_RANK>,

    weight: Vec<T>,
    bias: Vec<T>,
}

impl<T: NumT, const INPUT_RANK: usize, const OUTPUT_RANK: usize>
    Layer<T, INPUT_RANK, T, OUTPUT_RANK> for Dense<T, INPUT_RANK, OUTPUT_RANK>
{
    fn predict(&self, input: &Tensor<T, INPUT_RANK>) -> Result<Tensor<T, OUTPUT_RANK>> {
        if input.shape != self.input_shape {
            return Err(ShapeMismatchError);
        }
        Err(ShapeMismatchError)
    }
}