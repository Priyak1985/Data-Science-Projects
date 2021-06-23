<h1> Introduction</h1>

This repository hosts a collection of data science projects completed by me as a part of either independent initiatives or certification programs. The process of coding for the project, in turn, has honed my problem solving and coding skills. The notebooks hosted in the repository are ready to be run and reused by fellow coders.

Going deeper: convolutional autoencoder
PCA is neat but surely we can do better. This time we want you to build a deep convolutional autoencoder by... stacking more layers.

Encoder
The encoder part is pretty standard, we stack convolutional and pooling layers and finish with a dense layer to get the representation of desirable size (code_size).

We recommend to use activation='elu' for all convolutional and dense layers.

We recommend to repeat (conv, pool) 4 times with kernel size (3, 3), padding='same' and the following numbers of output channels: 32, 64, 128, 256.

Remember to flatten (L.Flatten()) output before adding the last dense layer!

Decoder
For decoder we will use so-called "transpose convolution".

Traditional convolutional layer takes a patch of an image and produces a number (patch -> number). In "transpose convolution" we want to take a number and produce a patch of an image (number -> patch). We need this layer to "undo" convolutions in encoder. We had a glimpse of it during week 3 (watch this video starting at 5:41).

Here's how "transpose convolution" works:
