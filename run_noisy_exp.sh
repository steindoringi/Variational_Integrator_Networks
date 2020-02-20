#!/bin/bash

for seed in 1
do
  for observations in noisy
  do
    for num_traj in 1 3 5
    do
      for model_name in ResNet VIN_VV
      do
        python run_exp.py ./experiments $model_name pendulum $observations $num_traj 21 $seed
      done
    done
  done
done