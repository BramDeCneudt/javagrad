package com.axxes.javagrad;

import com.axxes.javagrad.network.Network;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.util.ArrayList;
import java.util.List;

@SpringBootApplication
@Slf4j
public class JavagradCommandLineApplication implements CommandLineRunner {

	public static void main(String[] args) {
		SpringApplication.run(JavagradCommandLineApplication.class, args);
	}

    @Override
    public void run(String... args) throws Exception {
        // Load MNIST training data
        int batchSize = 1; // Load one sample at a time for conversion
        boolean train = true; // Use training data
        MnistDataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, train, 12345);

        // Convert DataSet to List<INDArray[]> format
        List<INDArray[]> trainingData = new ArrayList<>();
        while (mnistTrain.hasNext()) {
            DataSet ds = mnistTrain.next();
            INDArray features = ds.getFeatures(); // 784 pixels (28x28)
            INDArray labels = ds.getLabels(); // 10 one-hot encoded labels

            // Reshape features to column vector (784, 1)
            features = features.reshape(784, 1);
            // Reshape labels to column vector (10, 1)
            labels = labels.reshape(10, 1);

            trainingData.add(new INDArray[]{features, labels});
        }

        log.info("Loaded {} training samples", trainingData.size());

        var network = new Network(784, 30, 10);
        network.SGD(trainingData, 30, 10, 3.0);
    }
}
