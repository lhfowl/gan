export EXPERIMENT_NAME='imagenet128_baseline'
# export EXPERIMENT_NAME='imagenet128_expt_1ctd'
export OUTPUT_DIR=/scratch0/ilya/locDoc/tfgan/${EXPERIMENT_NAME}
export DATA_DIR=/scratch1/ilya/locDoc/data/tfdf

export BATCH_SIZE=128
export DATASET_ARGS='--image_size=128 --dataset_name=imagenet2012 --num_classes=1000 --dataset_val_split_name=validation'

export ADDITIONAL='--critic_type=acgan \
--dont_load_real_data=true \
--num_eval_steps=2'

bash gpu/_eval_base.sh