# export EXPERIMENT_NAME='imagenet64_baseline'
# export EXPERIMENT_NAME='imagenet64_fixed'
# export EXPERIMENT_NAME=imagenet64_acbaseline_2step
export EXPERIMENT_NAME=imagenet64_baseline_ctd
export OUTPUT_DIR=/scratch0/ilya/locDoc/tfgan/${EXPERIMENT_NAME}
export DATA_DIR=/scratch1/ilya/locDoc/data/tfdf

export BATCH_SIZE=64
export DATASET_ARGS='--image_size=64 --dataset_name=imagenet_resized/64x64 --num_classes=1000 --dataset_val_split_name=validation'

export ADDITIONAL='--critic_type=acgan \
--mode=gen_images \
--n_images_per_side_to_gen_per_class=8'

bash gpu/_eval_base.sh
