#!/bin/bash

#Download Data
cd data

#ELMO
mkdir elmo
cd elmo

##Original size
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json
mv elmo_2x4096_512_2048cnn_2xhighway_options.json 2x4096_512_2048cnn_2xhighway_options.json
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5
mv elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5 2x4096_512_2048cnn_2xhighway_weights.hdf5

##Medium size
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5
mv elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5 2x2048_256_2048cnn_1xhighway_weights.hdf5
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json
mv elmo_2x2048_256_2048cnn_1xhighway_options.json 2x2048_256_2048cnn_1xhighway_options.json

##Small size
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5
mv elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5 2x1024_128_2048cnn_1xhighway_weights.hdf5
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json
mv elmo_2x1024_128_2048cnn_1xhighway_options.json 2x1024_128_2048cnn_1xhighway_options.json

#Glove
cd ..
mkdir glove
cd glove
wget https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz