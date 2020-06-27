
export EXPERIMENT_NAME=cirfar100_kp1_wgan
export BATCH_SIZE=64
export TRAIN_STEPS_PER_EVAL=2500
export DATASET_ARGS='--image_size=32 --dataset_name=cifar100 --num_classes=100 --dataset_val_split_name=test'

# for k+1 mhinge with fm
export ADDITIONAL='--critic_type=discriminator_32_kplusone_wgan \
--generator_loss_fn=kplusone_wasserstein_generator_loss \
--kplusone_mhinge_cond_discriminator_weight=1.0 \
--aux_mhinge_cond_generator_weight=0.05 \
--tpu_gan_estimator_d_step=4 \
--extra_eval_metrics \
--eval_batch_size=1024 \
--num_eval_steps=9 \
--keep_checkpoint_max=20'

bash tpu/_base.sh