#!/bin/bash/

nohup python train_fintune_summarize.py  > run.log 2>&1 &


nohup python train_reward_model.py  > run2.log 2>&1 &
