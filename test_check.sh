#!/bin/bash

# Copyright 2014&2015 dpxlbx

# Train neural network
train_tool="./src/nnetbin/cnn_train_mGPU"
dir="."

# Training Data Location
feats_tr="/home/maohz/kalditu/egs/200k/s5/data/train/feats_total.scp"
feats_cv="/home/maohz/kalditu/egs/200k/s5/data/cv/feats_total.scp"
labels_tr="/home/maohz/kalditu/egs/200k/s5/exp/ali/train.dat"
labels_cv="/home/maohz/kalditu/egs/200k/s5/exp/ali/cv.dat"

# NN Parameter
mlp_init="nnet_dnn_test.init"
feature_transform="/home/maohz/kaldi-trunk/egs/200k/s5/exp/final.feature_transform"

# Training Options
learn_rate=0.00
minibatch_size=1024
randomizer_size=32768

# Learn Rate Scheduling
max_iters=20
# absolute decrease value for halving learn rate
start_halving_impr=0.5
# absolute decrease value for halving learn rate
end_halving_impr=0.1
halving_factor=0.5
 
# End configuration.

[ ! -d $dir ] && mkdir $dir
[ ! -d $dir/log ] && mkdir $dir/log
[ ! -d $dir/nnet ] && mkdir $dir/nnet

# Skip training
[ -e $dir/final.nnet ] && echo "'$dir/final.nnet' exists, skipping training" && exit 0

##############################
#start training

# choose mlp to start with
mlp_best=$mlp_init
mlp_base=${mlp_init##*/}; mlp_base=${mlp_base%.*}

# cross-validation on original network
log=$dir/log/prerun.log
$train_tool --num-threads=1 --num-updates=1 --cross-validate=true --learn-rate=$learn_rate --bunchsize=$minibatch_size \
            --cachesize=$randomizer_size --feature-transform=$feature_transform \
            $mlp_best $feats_cv $labels_cv \
            > $log || exit 1;
#$train_tool 1 $feats_cv $labels_cv $feature_transform \
# $learn_rate $minibatch_size $randomizer_size \
# $mlp_best \
# > $log || exit 1;
 
accuracy=$(cat $log | grep "FRAME_ACCURACY >>" | tail -n 1 | awk '{printf substr($3,1,length($3)-1)}')
accuracy_type=FRAME_ACCURACY
echo "CROSSVAL PRERUN AVG.Accuracy $(printf "%.4f" $accuracy) $accuracy_type"