package com.axxes.javagrad;

import com.axxes.javagrad.network.TestNetwork;
import com.axxes.javagrad.network.Trainer;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@Slf4j
public class JavagradCommandLineApplication implements CommandLineRunner {

	public static void main(String[] args) {
		SpringApplication.run(JavagradCommandLineApplication.class, args);
	}

    @Override
    public void run(String... args) throws Exception {
        log.info("Running JavagradCommandLineApplication");

        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized

        TestNetwork testNetwork = TestNetwork.builder()
                .statsStorage(statsStorage)
                .build();
        var model = testNetwork.generateNetwork();


        Trainer trainer = Trainer.builder()
                .model(model)
                .build();
        trainer.init();
        trainer.train();
        trainer.evaluate();
    }
}
