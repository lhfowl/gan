# For this experiment compare steps 580000 and 585000.
# The model was trained like a normal SAGAN until 580k steps.
# Then the model was trained according to the high fidelity low diversity
# method described in the paper.

export EXPERIMENT_NAME=imagenet128_baseline_ctd
export OUTPUT_DIR=/scratch0/ilya/locDoc/tfgan/examples/${EXPERIMENT_NAME}
export DATA_DIR=/scratch1/ilya/locDoc/data/tfdf

export BATCH_SIZE=36
# For generating images the dataset it not used, here we save the trouble of
# downloading imagenet128 by employing the much smaller imagenet_resize as a
# vacuous placeholder.
export DATASET_ARGS='--image_size=128 --dataset_name=imagenet_resized/64x64 --num_classes=1000 --dataset_val_split_name=validation'

# without gen_images_uniform_random_labels==True one tile of images per class
# will be generated. By default 1000 tiles of size BATCH_SIZE will be generated.
export ADDITIONAL='--critic_type=acgan_multiproj \
--mode=gen_images \
--gen_images_with_margins \
--n_images_per_side_to_gen_per_tile=6'

bash gpu/_eval_base.sh
