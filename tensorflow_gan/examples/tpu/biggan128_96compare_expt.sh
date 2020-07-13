
export EXPERIMENT_NAME=imagenet128_biggan_expt96
export BATCH_SIZE=1024
export TRAIN_STEPS_PER_EVAL=2500
export DATASET_ARGS='--image_size=128 --dataset_name=imagenet2012 --num_classes=1000 --dataset_val_split_name=validation'

export ADDITIONAL='--critic_type=acgan \
--z_dim=120 \
--tpu_gan_estimator_d_step=2 \
--discriminator_lr=0.0005 \
--tpu_iterations_per_loop=500 \
--gf_dim=96 \
--df_dim=96 \
--aux_mhinge_cond_generator_weight=0.1 \
--aux_mhinge_cond_discriminator_weight=1.0 \
--extra_eval_metrics=true \
--keep_checkpoint_max=10'

bash tpu/_base.sh