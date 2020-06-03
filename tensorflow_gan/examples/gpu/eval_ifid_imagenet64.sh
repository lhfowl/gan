
# export EXPERIMENT_NAME='imagenet64_fixed'
export EXPERIMENT_NAME='imagenet64_acbaseline_5step'
export OUTPUT_DIR=/scratch0/ilya/locDoc/tfgan/${EXPERIMENT_NAME}
export DATA_DIR=/scratch1/ilya/locDoc/data/tfdf

export BATCH_SIZE=128
export DATASET_ARGS='--image_size=64 --dataset_name=imagenet_resized/64x64 --num_classes=1000 --dataset_val_split_name=train'

# export ADDITIONAL='--critic_type=acgan \
# --num_eval_steps=40 \
# --mode=continuous_eval \
# --extra_eval_metrics'

#156 for 1k
export ADDITIONAL="--critic_type=acgan \
--num_eval_steps=390 \
--mode=intra_fid_eval \
--intra_fid_eval_chunk_size=20"

# export ADDITIONAL='--critic_type=acgan_multiproj \
# --num_eval_steps=390 \
# --mode=continuous_eval \
# --extra_eval_metrics'

bash gpu/_eval_base.sh
