
# export EXPERIMENT_NAME=imagenet128_baseline
export EXPERIMENT_NAME=imagenet128_expt_3ctd_bkp795

export OUTPUT_DIR=/scratch0/ilya/locDoc/tfgan/${EXPERIMENT_NAME}
export DATA_DIR=/scratch1/ilya/locDoc/data/tfdf

export BATCH_SIZE=16
export DATASET_ARGS='--image_size=128 --dataset_name=imagenet_resized/64x64 --num_classes=1000 --dataset_val_split_name=validation'

export ADDITIONAL='--critic_type=acgan \
--mode=gen_images \
--gen_images_with_margins \
--n_images_per_side_to_gen_per_class=4 \
--aux_mhinge_cond_generator_weight=0.05 \
--aux_mhinge_cond_discriminator_weight=1.0'

# export ADDITIONAL='--critic_type=acgan_multiproj \
# --mode=gen_images \
# --gen_images_with_margins \
# --n_images_per_side_to_gen_per_class=4'

bash gpu/_eval_base.sh
