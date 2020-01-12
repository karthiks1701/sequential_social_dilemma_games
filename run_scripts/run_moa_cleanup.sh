#!/usr/bin/env bash

python train.py \
--exp_name cleanup_moa \
--env cleanup \
--model moa \
--algorithm A3C \
--num_agents 5 \
--sample_batch_size 1000 \
--train_batch_size 30000 \
--stop_at_timesteps_total $((500 * 10 ** 6)) \
--memory $((50 * 10 ** 9)) \
--num_workers 12 \
--num_cpus_per_worker 1 \
--num_gpus_per_worker 0.25 \
--num_gpus_for_driver 1 \
--num_cpus_for_driver 1 \
--num_samples 1 \
--lr_schedule_steps 0 20000000 \
--lr_schedule_weights 0.00126 0.000012 \
--entropy_coeff 0.00176 \
--aux_loss_weight 0.06663557 \
--aux_reward_weight 1.0 \
--aux_reward_schedule_steps 0 10000000 100000000 300000000 \
--aux_reward_schedule_weights 0.0 0.0 1.0 0.5