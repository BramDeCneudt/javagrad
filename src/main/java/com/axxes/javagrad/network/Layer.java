package com.axxes.javagrad.network;

import java.util.ArrayList;
import java.util.List;

public record Layer(List<Neuron> neurons) {

    public Layer() {
        this(new ArrayList<>());
    }

}
