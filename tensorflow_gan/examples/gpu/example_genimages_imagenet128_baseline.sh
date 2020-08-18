# TESTIN COMPATIBILITY ONLY DO NOT USE!
# WRONG DATASET just for compatibility testing purposes

export EXPERIMENT_NAME='imagenet128_baseline'
export OUTPUT_DIR=/home/lfowl/Desktop/sagan_imgs/${EXPERIMENT_NAME}
export DATA_DIR=/home/lfowl/data

export BATCH_SIZE=16
export DATASET_ARGS='--image_size=128 --dataset_name=imagenet_resized/64x64 --num_classes=1000 --dataset_val_split_name=validation'


export ADDITIONAL='--critic_type=acgan_multiproj \
--mode=gen_images \
--n_images_per_side_to_gen_per_tile=1'

bash gpu/_eval_base.sh
