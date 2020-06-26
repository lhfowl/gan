
export EXPERIMENT_NAME=cifar100_baseline
export BATCH_SIZE=64
export TRAIN_STEPS_PER_EVAL=10000
export DATASET_ARGS='--image_size=32 --dataset_name=cifar100 --num_classes=100 --dataset_val_split_name=test'


export ADDITIONAL='--critic_type=acgan \
--extra_eval_metrics \
--eval_batch_size=1024 \
--num_eval_steps=9'

bash tpu/_base.sh