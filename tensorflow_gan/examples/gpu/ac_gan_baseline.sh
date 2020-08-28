# TESTIN COMPATIBILITY ONLY DO NOT USE!
# WRONG DATASET just for compatibility testing purposes

export EXPERIMENT_NAME='imagenet128_acbaseline'
export OUTPUT_DIR=/home/lfowl/Desktop/SAGAN/${EXPERIMENT_NAME}
export DATA_DIR=/home/lfowl/data

export BATCH_SIZE=16
export DATASET_ARGS='--image_size=128 --dataset_name=imagenet_resized/8x8 --num_classes=4000 --dataset_val_split_name=validation'


export ADDITIONAL='--critic_type=acgan \
--mode=gen_images \
--n_images_per_side_to_gen_per_tile=1'

bash gpu/_eval_base.sh
