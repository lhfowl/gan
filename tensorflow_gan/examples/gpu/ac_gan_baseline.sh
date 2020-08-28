# TESTIN COMPATIBILITY ONLY DO NOT USE!
# WRONG DATASET just for compatibility testing purposes

export EXPERIMENT_NAME='imagenet128_acbaseline'
export OUTPUT_DIR=/scratch0/ilya/locDoc/tfgan/examples/${EXPERIMENT_NAME}
export DATA_DIR=/scratch1/ilya/locDoc/data/tfdf

export BATCH_SIZE=16
export DATASET_ARGS='--image_size=128 --dataset_name=imagenet2012 --num_classes=1000 --dataset_val_split_name=validation'


export ADDITIONAL='--critic_type=acgan \
--num_eval_steps=2 \
--mode=continuous_eval'

bash gpu/_eval_base.sh
