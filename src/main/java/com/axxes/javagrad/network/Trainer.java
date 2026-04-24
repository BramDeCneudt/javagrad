package com.axxes.javagrad.network;

import lombok.Builder;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
@Builder
public class Trainer {

    @Builder.Default
    private final int epochs = 30;
    @Builder.Default
    private final int mini_batch_size = 10;
}
