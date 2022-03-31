extern crate easynn;
extern crate rust_mnist;

use easynn::prelude::*;
use rust_mnist::{print_sample_image, Mnist};

fn get_data(path: &str) -> (Vec::<Tensor<f64>>, Vec::<Tensor<f64>>, Vec::<Tensor<f64>>, Vec::<Tensor<f64>>) {
    // create the training set
    let mnist = Mnist::new(path);
    // Print one image (the one at index 5) for verification.
    print_sample_image(&mnist.train_data[5], mnist.train_labels[5]);
    let mut train_ims = Vec::<Tensor<f64>>::new();      // images
    let mut train_lbs = Vec::<Tensor<f64>>::new();      // one-hot classifications
    let mut test_ims = Vec::<Tensor<f64>>::new();      // images
    let mut test_lbs = Vec::<Tensor<f64>>::new();      // one-hot classifications
    for im in mnist.train_data {
        let mut im_f = Vec::<f64>::new();
        for x in im {
            im_f.push(x.into());
        }
        train_ims.push(Tensor::new(sh!([28, 28]), im_f));
    }
    for cl in mnist.train_labels {
        let mut cl_hot = Tensor::<f64>::zeros(sh!([10]));
        cl_hot.set([cl as usize], 1.);
        train_lbs.push(cl_hot);
    }
    for im in mnist.test_data {
        let mut im_f = Vec::<f64>::new();
        for x in im {
            im_f.push(x.into());
        }
        test_ims.push(Tensor::new(sh!([28, 28]), im_f));
    }
    for cl in mnist.test_labels {
        let mut cl_hot = Tensor::<f64>::zeros(sh!([10]));
        cl_hot.set([cl as usize], 1.);
        test_lbs.push(cl_hot);
    }
    (train_ims, train_lbs, test_ims, test_lbs)
}

fn main() {
    // create the network and add 4 layers
    let mut nn = Sequential::<f64>::new(Loss::MeanSquare);
    // add the input (2 integers) and a hidden layer
    nn.add(Dense::<f64>::new(sh!([28, 28]), sh!([256]), Activation::Relu));
    // add another hidden layer
    nn.add(Dense::<f64>::new(sh!([256]), sh!([128]), Activation::Relu));
    // add another hidden layer
    nn.add(Dense::<f64>::new(sh!([128]), sh!([64]), Activation::Relu));
    // add the output layer
    nn.add(Dense::<f64>::new(sh!([64]), sh!([10]), Activation::Relu));

    // Please download the dataset to the directory
    let (train_ims, train_lbs, test_ims, test_lbs) = get_data("../data/FashionMNIST/raw/");

    // train the model
    for e in 0..2 {
        println!("[Epoch {}]", e);
        nn.train_once(&train_ims, &train_lbs, 512, 0.01, true);
    }

    // evaluate the model
    let mut loss = 0.;
    for (im, lb) in test_ims.iter().zip(test_lbs.iter()) {
        let pred_lb = nn.predict(&im).unwrap();
        loss += nn.loss.call(&lb, &pred_lb).unwrap();
    }
    loss /= test_ims.len() as f64;
    println!("The final test average loss is {}", loss);
}