
export EXPERIMENT_NAME=imagenet128_baseline_96_bs2048
export BATCH_SIZE=1024
export TRAIN_STEPS_PER_EVAL=5000
export DATASET_ARGS='--image_size=128 --dataset_name=imagenet2012 --num_classes=1000 --dataset_val_split_name=validation'

export ADDITIONAL='--critic_type=acgan \
--tpu_gan_estimator_d_step=2 \
--tpu_gan_estimator_g_step=2 \
--gf_dim=96 \
--df_dim=96'

bash tpu/_base.sh