
export EXPERIMENT_NAME=cifar100_baseline
export BATCH_SIZE=1024
export TRAIN_STEPS_PER_EVAL=10000
export DATASET_ARGS='--image_size=32 \
--dataset_name=cifar100 \
--num_classes=100 \
--dataset_val_split_name=train'
export ADDITIONAL="--critic_type=acgan \
--num_eval_steps=100 \
--mode=intra_fid_eval \
--intra_fid_eval_chunk_size=50"

bash tpu/_eval_base.sh

