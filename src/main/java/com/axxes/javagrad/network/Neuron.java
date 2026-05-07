package com.axxes.javagrad.network;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

public record Neuron(
        Double activation,
        Double biases,
        double[] weights
) {

}
