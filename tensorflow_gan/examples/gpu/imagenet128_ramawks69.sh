export EXPERIMENT_NAME=test
export OUTPUT_DIR=/scratch0/ilya/locDoc/tfgan/${EXPERIMENT_NAME}
export DATA_DIR=/scratch1/ilya/locDoc/data/tfdf

export BATCH_SIZE=128
export TRAIN_STEPS_PER_EVAL=10000
export DATASET_ARGS='--image_size=128 \
--dataset_name=imagenet_resized/64x64 \
--num_classes=1000 \
--dataset_val_split_name=validation'
export ADDITIONAL='--critic_type=biggan_acgan \
--aux_mhinge_cond_generator_weight=0.1 \
--aux_mhinge_cond_discriminator_weight=1.0 \
--z_dim=120'

bash gpu/_base.sh