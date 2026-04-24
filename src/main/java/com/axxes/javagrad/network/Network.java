package com.axxes.javagrad.network;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

@Slf4j
public class Network {

    /*
    TODO - I have omitted the test data function for now
     */

    private int numLayers;
    private int[] sizes;
    private List<INDArray> biases;
    private List<INDArray> weights;

    // sizes: number of neurons per layer, e.g. {784, 30, 10}
    // input layer has no biases, so we start from index 1
    public Network(int... sizes) {
        this.numLayers = sizes.length;
        this.sizes = sizes;

        // one bias column vector per layer (excluding input layer)
        // e.g. for {784, 30, 10}: shapes (30,1) and (10,1)
        this.biases = new ArrayList<>();
        for (int i = 1; i < sizes.length; i++) {
            biases.add(Nd4j.randn(sizes[i], 1));
        }

        // one weight matrix per connection between adjacent layers
        // shape is (next layer size, current layer size)
        // e.g. for {784, 30, 10}: shapes (30,784) and (10,30)
        // here we init the weights BETWEEN the layers, so it starts with plus 1 and ends BEFORE the last one
        this.weights = new ArrayList<>();
        for (int i = 0; i < sizes.length - 1; i++) {
            weights.add(Nd4j.randn(sizes[i + 1], sizes[i]));
        }
    }


    // trainingData: list of (x, y) pairs
// epochs: number of full passes through the training data
// miniBatchSize: number of samples per mini-batch
// eta: learning rate
    public void SGD(List<INDArray[]> trainingData, int epochs, int miniBatchSize, double eta) {
        int n = trainingData.size();

        for (int j = 0; j < epochs; j++) {
            long time1 = System.currentTimeMillis();

            // shuffle training data to randomize mini-batch composition each epoch
            Collections.shuffle(trainingData);

            // split training data into mini-batches of size miniBatchSize
            List<List<INDArray[]>> miniBatches = new ArrayList<>();
            for (int k = 0; k < n; k += miniBatchSize) {
                miniBatches.add(trainingData.subList(k, Math.min(k + miniBatchSize, n)));
            }

            // update weights and biases for each mini-batch
            for (List<INDArray[]> miniBatch : miniBatches) {
                updateMiniBatch(miniBatch, eta);
            }

            long time2 = System.currentTimeMillis();
            System.out.printf("Epoch %d complete in %.2f seconds%n", j, (time2 - time1) / 1000.0);
        }
    }

    // mini_batch: list of (x, y) pairs where x is input and y is expected output
// eta: learning rate
    public void updateMiniBatch(List<INDArray[]> miniBatch, double eta) {

        // initialize gradient accumulators to zero, one matrix per layer
        List<INDArray> nablaB = new ArrayList<>();
        List<INDArray> nablaW = new ArrayList<>();
        for (INDArray b : biases) {
            nablaB.add(Nd4j.zeros(b.shape()));
        }
        for (INDArray w : weights) {
            nablaW.add(Nd4j.zeros(w.shape()));
        }

        // accumulate gradients over all samples in the mini-batch
        for (INDArray[] sample : miniBatch) {
            INDArray x = sample[0];
            INDArray y = sample[1];

            // backprop returns the gradient for this single sample
            List<INDArray>[] deltas = backprop(x, y);
            List<INDArray> deltaNablaB = deltas[0];
            List<INDArray> deltaNablaW = deltas[1];

            // add this sample's gradient to the running total
            for (int i = 0; i < nablaB.size(); i++) {
                nablaB.get(i).addi(deltaNablaB.get(i));
            }
            for (int i = 0; i < nablaW.size(); i++) {
                nablaW.get(i).addi(deltaNablaW.get(i));
            }
        }

        // update weights and biases: w = w - (eta / batchSize) * nablaW
        double scale = eta / miniBatch.size();
        for (int i = 0; i < weights.size(); i++) {
            weights.set(i, weights.get(i).sub(nablaW.get(i).mul(scale)));
        }
        for (int i = 0; i < biases.size(); i++) {
            biases.set(i, biases.get(i).sub(nablaB.get(i).mul(scale)));
        }
    }


    // returns [nablaB, nablaW] — gradients of the cost function for a single sample (x, y)
    @SuppressWarnings("unchecked")
    public List<INDArray>[] backprop(INDArray x, INDArray y) {

        // initialize gradient accumulators to zero
        List<INDArray> nablaB = new ArrayList<>();
        List<INDArray> nablaW = new ArrayList<>();
        for (INDArray b : biases) nablaB.add(Nd4j.zeros(b.shape()));
        for (INDArray w : weights) nablaW.add(Nd4j.zeros(w.shape()));

        // --- feedforward pass ---
        // store all activations and z vectors layer by layer
        INDArray activation = x;
        List<INDArray> activations = new ArrayList<>();
        activations.add(x);
        List<INDArray> zs = new ArrayList<>();

        for (int i = 0; i < biases.size(); i++) {
            // z = w * activation + b
            INDArray z = weights.get(i).mmul(activation).add(biases.get(i));
            zs.add(z);
            activation = sigmoid(z);
            activations.add(activation);
        }

        // --- backward pass ---
        // compute error at the output layer: delta = cost_derivative * sigmoid_prime(z)
        int last = zs.size() - 1;
        INDArray delta = costDerivative(activations.getLast(), y)
                .mul(sigmoidPrime(zs.get(last)));
        nablaB.set(last, delta);
        // nablaW for output layer = delta * activation_of_previous_layer^T
        nablaW.set(last, delta.mmul(activations.get(activations.size() - 2).transpose()));

        // propagate error backwards through hidden layers
        // l=1 is last layer, l=2 is second-last, etc. (matches Python's negative indexing)
        for (int l = 2; l < numLayers; l++) {
            INDArray z = zs.get(zs.size() - l);
            INDArray sp = sigmoidPrime(z);
            // delta = (w of next layer)^T * delta * sigmoid_prime(z)
            delta = weights.get(weights.size() - l + 1).transpose().mmul(delta).mul(sp);
            nablaB.set(nablaB.size() - l, delta);
            nablaW.set(nablaW.size() - l, delta.mmul(activations.get(activations.size() - l - 1).transpose()));
        }

        return new List[]{nablaB, nablaW};
    }

    // element-wise sigmoid: 1 / (1 + e^-z)
    private INDArray sigmoid(INDArray z) {
        return Transforms.sigmoid(z);
    }

    // derivative of sigmoid: sigmoid(z) * (1 - sigmoid(z))
    private INDArray sigmoidPrime(INDArray z) {
        INDArray s = sigmoid(z);
        return s.mul(s.rsub(1.0));
    }

    // derivative of cost function: output - expected
    private INDArray costDerivative(INDArray outputActivations, INDArray y) {
        return outputActivations.sub(y);
    }


}
