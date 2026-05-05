package com.axxes.javagrad;

import com.axxes.javagrad.network.TestNetwork;
import com.axxes.javagrad.network.Trainer;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;

@Slf4j
public class MainApplication {

    public static void main(String[] args) throws Exception {
        log.info("Running JavagradCommandLineApplication");

        // Initialize UI Server with explicit configuration
        log.info("Starting UI Server...");
        UIServer uiServer = UIServer.getInstance();

        // Wait a bit for the server to fully start
        Thread.sleep(2000);

        log.info("UI Server is running at: http://localhost:9000");

        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);

        TestNetwork testNetwork = TestNetwork.builder()
                .statsStorage(statsStorage)
                .build();
        var model = testNetwork.generateNetwork();

        Trainer trainer = Trainer.builder()
                .model(model)
                .build();
        trainer.init();
        model.fit();
//        trainer.train();
//        trainer.evaluate();
    }
}
