
# export BATCH_SIZE=64
# export ADDITIONAL='--critic_type=kplusone_fm --kplusone_mhinge_cond_discriminator_weight=1.0 --aux_mhinge_cond_generator_weight=0.05'
# export ADDITIONAL='--critic_type=kplusone_fm --generator_loss_fn=kplusone_ssl_featurematching_generator_loss --kplusone_mhinge_ssl_cond_discriminator_weight=1.0 --aux_mhinge_cond_generator_weight=0.05'

export EXPERIMENT_NAME=stl128
export BATCH_SIZE=1024
export TRAIN_STEPS_PER_EVAL=2000
export DATASET_ARGS='--image_size=128 \
--dataset_name=stl10 \
--num_classes=10 \
--dataset_val_split_name=test \
--unlabelled_dataset_name=stl10 \
--unlabelled_dataset_split_name=unlabelled'
export ADDITIONAL='--critic_type=kplusone_fm \
--generator_loss_fn=kplusone_ssl_featurematching_generator_loss \
--kplusone_mhinge_ssl_cond_discriminator_weight=1.0 \
--aux_mhinge_cond_generator_weight=0.05 \
--tpu_gan_estimator_d_step=4'

bash tpu/_base.sh