package com.axxes.javagrad.network.listener;

import com.axxes.javagrad.network.Epoch;
import com.axxes.javagrad.network.Network;
import com.axxes.javagrad.network.Neuron;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

@Slf4j
public class ModelListener extends BaseTrainingListener {

    private final Network network = new Network(List.of(784, 10, 10));

    @Override
    public void onEpochEnd(Model model) {

        Epoch epoch = new Epoch();
        network.epoches().add(epoch);

        log.info("Score: {}", model.score());

        MultiLayerNetwork net = (MultiLayerNetwork) model;
        Layer[] layers = net.getLayers();

        // run one forward pass to get activations for the sample input
        List<INDArray> activations = net.feedForward();

        for (int i = 0; i < layers.length; i++) {
            com.axxes.javagrad.network.Layer layer = new com.axxes.javagrad.network.Layer();
            epoch.layers().add(layer);

            INDArray weights = layers[i].getParam("W"); // shape: [nIn, nOut]
            INDArray biases  = layers[i].getParam("b"); // shape: [1, nOut]
            INDArray layerActivations = activations.get(i + 1); // shape: [1, nOut]

            int numNeurons = (int) weights.size(1);
            log.info("Layer {}: {} neurons", i, numNeurons);

            for (int neuronIndex = 0; neuronIndex < numNeurons; neuronIndex++) {
                INDArray neuronWeights = weights.getColumn(neuronIndex);
                double neuronBias = biases.getDouble(neuronIndex);
                double neuronActivation = layerActivations.getDouble(neuronIndex);
                log.info("layer={}  Neuron {}: bias={}, activation={}, weights={}", i, neuronIndex, neuronBias, neuronActivation, neuronWeights);

                Neuron neuron = new Neuron(neuronActivation, neuronBias, neuronWeights.toDoubleVector());
                layer.neurons().add(neuron);
            }
        }
        network.write();
    }
}
