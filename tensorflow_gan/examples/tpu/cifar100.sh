
export EXPERIMENT_NAME=cifar100_expt
export BATCH_SIZE=64
export TRAIN_STEPS_PER_EVAL=10000
export DATASET_ARGS='--image_size=32 --dataset_name=cifar100 --num_classes=100 --dataset_val_split_name=test'

export ADDITIONAL='--critic_type=acgan \
--aux_mhinge_cond_generator_weight=0.1 \
--aux_mhinge_cond_discriminator_weight=1.0 \
--extra_eval_metrics \
--eval_batch_size=1024 \
--num_eval_steps=9 \
--keep_checkpoint_max=10'


bash tpu/_base.sh