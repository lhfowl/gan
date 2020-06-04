export EXPERIMENT_NAME='imagenet64_baseline'
# export EXPERIMENT_NAME='imagenet64_fixed'
export OUTPUT_DIR=/scratch0/ilya/locDoc/tfgan/${EXPERIMENT_NAME}
export DATA_DIR=/scratch1/ilya/locDoc/data/tfdf

export BATCH_SIZE=128
export DATASET_ARGS='--image_size=64 --dataset_name=imagenet_resized/64x64 --num_classes=1000 --dataset_val_split_name=validation'

export ADDITIONAL='--critic_type=acgan \
--num_eval_steps=8 \
--mode=intra_fid_eval'
# export ADDITIONAL='--critic_type=acgan \
# --num_eval_steps=80 \
# --mode=continuous_eval'

bash gpu/_eval_base.sh
