
# export BATCH_SIZE=64
# export ADDITIONAL='--critic_type=kplusone_fm --kplusone_mhinge_cond_discriminator_weight=1.0 --aux_mhinge_cond_generator_weight=0.05'
# export ADDITIONAL='--critic_type=kplusone_fm --generator_loss_fn=kplusone_ssl_featurematching_generator_loss --kplusone_mhinge_ssl_cond_discriminator_weight=1.0 --aux_mhinge_cond_generator_weight=0.05'
export EXPERIMENT_NAME=cifar_achinge1.0_withproj_marg0.0REALLY
export BATCH_SIZE=64
export TRAIN_STEPS_PER_EVAL=50000
export DATASET_ARGS='--image_size=32 --dataset_name=cifar10 --num_classes=10 --dataset_val_split_name=test'

# for k+1 mhinge with fm
# export ADDITIONAL='--critic_type=kplusone_fm \
# --generator_loss_fn=kplusone_featurematching_generator_loss \
# --kplusone_mhinge_cond_discriminator_weight=1.0 \
# --aux_mhinge_cond_generator_weight=0.05 \
# --tpu_gan_estimator_d_step=4'

export ADDITIONAL='--critic_type=acgan \
--aux_mhinge_cond_generator_weight=0.1 \
--aux_mhinge_cond_discriminator_weight=1.0 \
--generator_margin_size=0.0'

# cross entropy baseline
# export ADDITIONAL='--critic_type=acgan_noproj \
# --kplusone_mhinge_cond_discriminator_weight=1.0 \
# --aux_mhinge_cond_generator_weight=1.0 \
# --aux_mhinge_cond_discriminator_weight=1.0'

# worked earlier:
# export ADDITIONAL='--critic_type=acgan_noproj \
# --kplusone_mhinge_cond_discriminator_weight=1.0 \
# --aux_mhinge_cond_generator_weight=0.1 \
# --aux_mhinge_cond_discriminator_weight=1.0'

bash tpu/_base.sh