package com.axxes.javagrad.network;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.module.SimpleModule;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public record Network(List<Integer> layers, List<Epoch> epoches) {

    public Network(List<Integer> layers) {
        this(layers, new ArrayList<>());
    }

    public void write() {
        SimpleModule module = new SimpleModule();
        ObjectMapper mapper = new ObjectMapper().registerModule(module);
        try {
            Files.createDirectories(Path.of("./models"));
            mapper.writerWithDefaultPrettyPrinter()
                    .writeValue(Path.of("./models/network.json").toFile(), this);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
