
export EXPERIMENT_NAME=cifar100_acbaseline
export BATCH_SIZE=64
export TRAIN_STEPS_PER_EVAL=10000
export DATASET_ARGS='--image_size=32 --dataset_name=cifar100 --num_classes=100 --dataset_val_split_name=test'

# cross entropy baseline
export ADDITIONAL='--critic_type=acgan \
--aux_cond_generator_weight=0.1 \
--aux_cond_discriminator_weight=1.0 \
--extra_eval_metrics=true \
--tpu_gan_estimator_d_step=4'

bash tpu/_base.sh