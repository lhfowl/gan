
# export BATCH_SIZE=64
# export ADDITIONAL='--critic_type=kplusone_fm --kplusone_mhinge_cond_discriminator_weight=1.0 --aux_mhinge_cond_generator_weight=0.05'
# export ADDITIONAL='--critic_type=kplusone_fm --generator_loss_fn=kplusone_ssl_featurematching_generator_loss --kplusone_mhinge_ssl_cond_discriminator_weight=1.0 --aux_mhinge_cond_generator_weight=0.05'

export EXPERIMENT_NAME=ssl_imagenet128_withproj_baseline
export BATCH_SIZE=1024
export TRAIN_STEPS_PER_EVAL=10000
export DATASET_ARGS='--image_size=128 \
--dataset_name=imagenet2012_subset/10pct \
--num_classes=1000 \
--dataset_val_split_name=validation \
--unlabelled_dataset_name=imagenet2012 \
--unlabelled_dataset_split_name=train'
# export ADDITIONAL='--critic_type=kplusone_fm \
# --generator_loss_fn=kplusone_ssl_featurematching_generator_loss \
# --kplusone_mhinge_ssl_cond_discriminator_weight=1.0 \
# --aux_mhinge_cond_generator_weight=0.05 \
# --tpu_gan_estimator_d_step=4'

export ADDITIONAL='--critic_type=acgan \
--aux_cond_generator_weight=0.1 \
--aux_cond_discriminator_weight=1.0 \
--tpu_gan_estimator_d_step=2'

bash tpu/_base.sh