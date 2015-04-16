#!/bin/bash

# Copyright 2014&2015 dpxlbx

# Train neural network
train_tool="./cnn_feedforward"
dir="."
feats_in="990_in.dat"
pdf_out="9872_out.dat"
mlp_init="cnn_990.nnet"
size=50000

# Don't care
# Training Data Location
feats_cv="/home/maohz12/input_990_output_9872/data_new/cv.scp"
labels_cv="/home/maohz12/input_990_output_9872/data_new/cv.huizi.pdf"

# NN Parameter
feature_transform="/home/maohz12/input_990_output_9872/final_dnn.feature_transform"

# Training Options
learn_rate=0.00
minibatch_size=1024
randomizer_size=32768

# choose mlp to start with
mlp_best=$mlp_init

# cross-validation on original network
log=prerun.log
$train_tool --cross-validate=true --num-threads=1 --learn-rate=$learn_rate --bunchsize=$minibatch_size \
            --cachesize=$randomizer_size --feature-transform=$feature_transform \
            $mlp_best $feats_in $pdf_out $size\
            > $log || exit 1;
 
