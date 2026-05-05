package com.axxes.javagrad.network;

import lombok.Builder;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.Objects;

@Builder
@Slf4j
public class Trainer {

    @Builder.Default
    private final int batchSize = 64;
    @Builder.Default
    private final int seed = 123;
    @Builder.Default
    private final int epochs = 30;

    private final MultiLayerNetwork model;

    private DataSetIterator trainData;
    private DataSetIterator testData;

    @SneakyThrows
    public void init() {
        this.trainData = new MnistDataSetIterator(batchSize, true, seed);
        this.testData = new MnistDataSetIterator(batchSize, false, seed);
    }

    public void train() {
        Objects.requireNonNull(trainData, "trainData is null, init method was not called.");
        // Train
        for (int i = 0; i < epochs; i++) {
            model.fit(trainData);
            log.info("Completed epoch {}", i);
        }

    }

    public void evaluate() {
        Objects.requireNonNull(trainData, "trainData is null, init method was not called.");

        var eval = model.evaluate(testData);
        log.info(eval.stats());
    }
}
