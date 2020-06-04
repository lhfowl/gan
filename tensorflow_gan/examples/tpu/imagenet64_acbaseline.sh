
export EXPERIMENT_NAME=imagenet64_acbaseline_4step
export BATCH_SIZE=1024
export TRAIN_STEPS_PER_EVAL=10000
export DATASET_ARGS='--image_size=64 \
--dataset_name=imagenet_resized/64x64 \
--num_classes=1000 \
--dataset_val_split_name=validation'
export ADDITIONAL='--critic_type=acgan \
--aux_cond_generator_weight=0.1 \
--aux_cond_discriminator_weight=1.0 \
--tpu_gan_estimator_d_step=4'

# export EXPERIMENT_NAME=imagenet64_achinge0.1_noproj_marg1.0
# export BATCH_SIZE=1024
# export TRAIN_STEPS_PER_EVAL=10000
# export DATASET_ARGS='--image_size=64 --dataset_name=imagenet_resized/64x64 --num_classes=1000 --dataset_val_split_name=validation'

# export ADDITIONAL='--critic_type=acgan_noproj \
# --aux_mhinge_cond_generator_weight=0.1 \
# --aux_mhinge_cond_discriminator_weight=1.0 \
# --generator_margin_size=1.0'


bash tpu/_base.sh