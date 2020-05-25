export EXPERIMENT_NAME='testing'
export OUTPUT_DIR=/scratch0/ilya/locDoc/tfgan/${EXPERIMENT_NAME}
export DATA_DIR=/scratch0/ilya/locDoc/data/tfdf

export BATCH_SIZE=64
export DATASET_ARGS='--image_size=32 \
--dataset_name=stl10 \
--num_classes=10 \
--dataset_val_split_name=test \
--unlabelled_dataset_name=stl10 \
--unlabelled_dataset_split_name=unlabelled'
export ADDITIONAL='--critic_type=acgan \
--aux_mhinge_cond_generator_weight=0.1 \
--aux_mhinge_cond_discriminator_weight=1.0'


# export ADDITIONAL='--critic_type=kplusone_fm \
# --generator_loss_fn=kplusone_ssl_featurematching_generator_loss \
# --kplusone_mhinge_ssl_cond_discriminator_weight=1.0 \
# --aux_mhinge_cond_generator_weight=0.05'

# ssl
#   --unlabelled_dataset_name=stl10 --unlabelled_dataset_split_name=unlabelled \

bash gpu/_base.sh