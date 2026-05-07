package com.axxes.javagrad.network;

import java.util.ArrayList;
import java.util.List;

public record Epoch(
        List<Layer> layers
) {
    public Epoch() {
        this(new ArrayList<>());
    }

}
