# Usage:
# CUDA_VISIBLE_DEVICES=1 INTRA_FID_EVAL_START=125 INTRA_FID_EVAL_END=250 sh gpu/eval_imagenet64_tyan.sh
# CUDA_VISIBLE_DEVICES=2 INTRA_FID_EVAL_START=250 INTRA_FID_EVAL_END=375 sh gpu/eval_imagenet64_tyan.sh
# CUDA_VISIBLE_DEVICES=3 INTRA_FID_EVAL_START=375 INTRA_FID_EVAL_END=500 sh gpu/eval_imagenet64_tyan.sh

# export EXPERIMENT_NAME='imagenet64_baseline'
export EXPERIMENT_NAME='imagenet64_fixed'
export OUTPUT_DIR=/nfs/cluster/ilya/tfgan/${EXPERIMENT_NAME}
export DATA_DIR=/nfs/cluster/ilya/tfdf

export BATCH_SIZE=128
export DATASET_ARGS='--image_size=64 --dataset_name=imagenet_resized/64x64 --num_classes=1000 --dataset_val_split_name=validation'

export ADDITIONAL="--critic_type=acgan \
--num_eval_steps=8 \
--mode=intra_fid_eval \
--tfdf_num_parallel_calls=2 \
--intra_fid_eval_start=${INTRA_FID_EVAL_START} \
--intra_fid_eval_end=${INTRA_FID_EVAL_END}"

# export ADDITIONAL='--critic_type=acgan \
# --num_eval_steps=80 \
# --mode=continuous_eval'

bash gpu/_eval_base.sh