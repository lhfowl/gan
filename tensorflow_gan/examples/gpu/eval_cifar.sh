export EXPERIMENT_NAME='imagenet64_baseline'
# export EXPERIMENT_NAME='imagenet64_fixed'
export OUTPUT_DIR=/scratch0/ilya/locDoc/tfgan/${EXPERIMENT_NAME}
export DATA_DIR=/scratch1/ilya/locDoc/data/tfdf

export BATCH_SIZE=64
export DATASET_ARGS='--image_size=32 --dataset_name=cifar10 --num_classes=10 --dataset_val_split_name=test'
# export ADDITIONAL='--critic_type=acgan \
# --num_eval_steps=40 \
# --mode=continuous_eval \
# --extra_eval_metrics'


export ADDITIONAL='--critic_type=acgan_multiproj \
--num_eval_steps=40 \
--mode=continuous_eval \
--extra_eval_metrics'

bash gpu/_eval_base.sh
