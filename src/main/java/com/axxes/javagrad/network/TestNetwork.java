package com.axxes.javagrad.network;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.CheckpointListener;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

@Slf4j
@Builder
public class TestNetwork {

    @Builder.Default
    private final int seed = 123;
    @Builder.Default
    private final int inputLayer = 784;
    @Builder.Default
    private final int numHidden = 30;
    @Builder.Default
    private final int numOutputs = 10;
    private final StatsStorage statsStorage;

    public MultiLayerNetwork generateNetwork() {
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Sgd(0.1)) // similar to Nielsen's learning rate
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(inputLayer)
                        .nOut(numHidden)
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(numHidden)
                        .nOut(numOutputs)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

//        CheckpointListener.Builder builder = new CheckpointListener.Builder("./models/");
//        builder.saveEveryEpoch();

//        model.setListeners(builder.build());

        model.setListeners(new StatsListener(statsStorage));

        return model;
    }

}
