# TESTIN COMPATIBILITY ONLY DO NOT USE!
# WRONG DATASET just for compatibility testing purposes

export EXPERIMENT_NAME='imagenet128_expt'
export OUTPUT_DIR=/scratch0/ilya/locDoc/tfgan/examples/${EXPERIMENT_NAME}
export DATA_DIR=/scratch1/ilya/locDoc/data/tfdf

export BATCH_SIZE=16
export DATASET_ARGS='--image_size=128 --dataset_name=imagenet_resized/64x64 --num_classes=1000 --dataset_val_split_name=validation'

# include --gen_images_uniform_random_labels=True if you don't want 1 class
# per tile of images
export ADDITIONAL='--critic_type=acgan \
--mode=gen_images \
--n_images_per_side_to_gen_per_tile=4'


bash gpu/_eval_base.sh