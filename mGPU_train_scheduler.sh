#!/bin/bash

# Copyright 2014&2015 dpxlbx

# Train neural network
train_tool="./cnn_train_mGPU"
dir="."

# Training Data Location
feats_tr="/home/dpxlbx/kalditu/egs/200k/s5/data/train/feats_total.scp"
feats_cv="/home/dpxlbx/kalditu/egs/200k/s5/data/cv/feats_total.scp"
labels_tr="/home/dpxlbx/kalditu/egs/200k/s5/exp/ali/train.dat"
labels_cv="/home/dpxlbx/kalditu/egs/200k/s5/exp/ali/cv.dat"

# NN Parameter
mlp_init="cnn.init"
feature_transform="/home/dpxlbx/kalditu/lbx_nn_mGPU/final.feature_transform"

# Training Options
learn_rate=0.001
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
if [ ! -f $log ];then
$train_tool --cross-validate=true --learn-rate=$learn_rate --bunchsize=$minibatch_size \
            --cachesize=$randomizer_size --feature-transform=$feature_transform \
            $mlp_best $feats_cv $labels_cv \
            > $log || exit 1;
fi
#$train_tool 1 $feats_cv $labels_cv $feature_transform \
# $learn_rate $minibatch_size $randomizer_size \
# $mlp_best \
# > $log || exit 1;
 
accuracy=$(cat $log | grep "FRAME_ACCURACY >>" | tail -n 1 | awk '{printf substr($3,1,length($3)-1)}')
accuracy_type=FRAME_ACCURACY
echo "CROSSVAL PRERUN AVG.Accuracy $(printf "%.4f" $accuracy) $accuracy_type"

[ ! -f $dir/train.scp.10k ] && head -n 10000 $feats_tr > $dir/train.scp.10k
[ ! -f $dir/train.scp.1m ] && head -n 1000000 $feats_tr > $dir/train.scp.1m

# resume lr-halving
halving=0
# optionally resume training from the best epoch
[ -e $dir/.mlp_best ] && mlp_best=$(cat $dir/.mlp_best)
[ -e $dir/.learn_rate ] && learn_rate=$(cat $dir/.learn_rate)
[ -e $dir/.halving ] && halving=$(cat $dir/.halving)
# training
for iter in $(seq -w $max_iters); do
  echo -n "ITERATION $iter: "
  mlp_next=$dir/nnet/${mlp_base}_iter${iter}
  
  # skip iteration if already done
  [ -e $dir/.done_iter$iter ] && echo -n "skipping... " && ls $mlp_next* && continue 
  
  if [ $iter -le 1 ];then
    minibatch_size=256
    actual_feats_tr="$dir/train.scp.10k"
  elif [ $iter -le 2 ];then
    minibatch_size=256
    actual_feats_tr="$dir/train.scp.1m"
  else
    minibatch_size=1024
    actual_feats_tr=$feats_tr
  fi
  # training
  log=$dir/log/iter${iter}.tr.log
  $train_tool --num-updates=2 --num-threads=2 --cross-validate=false --learn-rate=$learn_rate --bunchsize=$minibatch_size \
              --cachesize=$randomizer_size --feature-transform=$feature_transform \
              $mlp_best $actual_feats_tr $labels_tr $mlp_next \
              > $log || exit 1;

  tr_accuracy=$(cat $dir/log/iter${iter}.tr.log | grep "FRAME_ACCURACY >>" | tail -n 1 | awk '{printf substr($3,1,length($3)-1)}')
  echo -n "TRAIN AVG.Accuracy $(printf "%.4f" $tr_accuracy), (lrate$(printf "%.6g" $learn_rate)), "
 
  # cross-validation
  log=$dir/log/iter${iter}.cv.log; hostname>$log
  $train_tool --cross-validate=true --learn-rate=$learn_rate --bunchsize=$minibatch_size \
              --cachesize=$randomizer_size --feature-transform=$feature_transform \
              $mlp_next $feats_cv $labels_cv \
   >$log || exit 1;
  
  accuracy_new=$(cat $dir/log/iter${iter}.cv.log | grep "FRAME_ACCURACY >>" | tail -n 1 | awk '{printf substr($3,1,length($3)-1)}')
  echo -n "CROSSVAL AVG.Accuracy $(printf "%.4f" $accuracy_new), "

  # accept or reject new parameters (based on objective function)
  accuracy_prev=$accuracy
  if [ "1" == "$(awk "BEGIN{print($accuracy_new>$accuracy);}")" ]; then
    accuracy=$accuracy_new
    mlp_best=$dir/nnet/${mlp_base}_iter${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_accuracy)_cv$(printf "%.4f" $accuracy_new)
    mv $mlp_next $mlp_best
    echo "nnet accepted ($(basename $mlp_best))"
    echo $mlp_best > $dir/.mlp_best 
  else
    mlp_reject=$dir/nnet/${mlp_base}_iter${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_accuracy)_cv$(printf "%.4f" $accuracy_new)_rejected
    mv $mlp_next $mlp_reject
    echo "nnet rejected ($(basename $mlp_reject))"
  fi

  # create .done file as a mark that iteration is over
  touch $dir/.done_iter$iter

  # stopping criterion
  if [[ "1" == "$halving" && "1" == "$(awk "BEGIN{print(($accuracy-$accuracy_prev) < $end_halving_impr)}")" ]]; then
    if [[ "$min_iters" != "" ]]; then
      if [ $min_iters -gt $iter ]; then
        echo we were supposed to finish, but we continue, min_iters : $min_iters
        continue
      fi
    fi
    echo finished, too small rel. improvement $(awk "BEGIN{print(($accuracy-$accuracy_prev))}")
    break
  fi

  # start annealing when improvement is low
  if [ "1" == "$(awk "BEGIN{print(($accuracy-$accuracy_prev) < $start_halving_impr)}")" ]; then
    halving=1
    echo $halving >$dir/.halving
  fi
  
  # do annealing
  if [ "1" == "$halving" ]; then
    learn_rate=$(awk "BEGIN{print($learn_rate*$halving_factor)}")
    echo $learn_rate >$dir/.learn_rate
  fi
done

# select the best network
if [ $mlp_best != $mlp_init ]; then 
  mlp_final=${mlp_best}_final_
  ( cd $dir/nnet; ln -s $(basename $mlp_best) $(basename $mlp_final); )
  ( cd $dir; ln -s nnet/$(basename $mlp_final) final.nnet; )
  echo "Succeeded training the Neural Network : $dir/final.nnet"
else
  "Error training neural network..."
  exit 1
fi
