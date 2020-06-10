
export EXPERIMENT_NAME='imagenet64_fixed'
export DATA_DIR=/scratch2/ilyak/locDoc/data/tfdf
export OUTPUT_DIR=/scratch2/ilyak/locDoc/tfgan/${EXPERIMENT_NAME}

export BATCH_SIZE=128
export DATASET_ARGS='--image_size=64 --dataset_name=imagenet_resized/64x64 --num_classes=1000 --dataset_val_split_name=train'

export ADDITIONAL="--critic_type=acgan \
--num_eval_steps=390 \
--mode=intra_fid_eval \
--intra_fid_eval_chunk_size=20"

bash gpu/_eval_base.sh