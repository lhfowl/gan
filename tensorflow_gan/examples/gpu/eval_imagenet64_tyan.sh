# export EXPERIMENT_NAME='imagenet64_baseline'
export EXPERIMENT_NAME='imagenet64_fixed'
export OUTPUT_DIR=/nfs/cluster/ilya/tfgan/${EXPERIMENT_NAME}
export DATA_DIR=/nfs/cluster/ilya/tfdf

export BATCH_SIZE=128
export DATASET_ARGS='--image_size=64 --dataset_name=imagenet_resized/64x64 --num_classes=1000 --dataset_val_split_name=validation'

export ADDITIONAL="--critic_type=acgan \
--num_eval_steps=8 \
--mode=intra_fid_eval \
--intra_fid_eval_start=${INTRA_FID_EVAL_START} \
--intra_fid_eval_end=${INTRA_FID_EVAL_END}"

# export ADDITIONAL='--critic_type=acgan \
# --num_eval_steps=80 \
# --mode=continuous_eval'

bash gpu/_eval_base.sh