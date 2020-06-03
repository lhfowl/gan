export EXPERIMENT_NAME='cirfar100_kp1'
export OUTPUT_DIR=/scratch0/ilya/locDoc/tfgan/${EXPERIMENT_NAME}
export DATA_DIR=/scratch0/ilya/locDoc/data/tfdf

export BATCH_SIZE=128
export DATASET_ARGS='--image_size=32 --dataset_name=cifar100 --num_classes=100 --dataset_val_split_name=train'

# discriminator doesn't matter
# but should be fm one
export ADDITIONAL="--critic_type=kplusone_wgan \
--generator_loss_fn=kplusone_featurematching_generator_loss \
--num_eval_steps=160 \
--mode=intra_fid_eval \
--intra_fid_eval_chunk_size=10"

bash gpu/_eval_base.sh
