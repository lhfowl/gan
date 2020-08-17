# For this experiment compare steps 999999 and 1014999.
# The model was trained like a normal SAGAN until 1M steps.
# Then the model was trained according to the high fidelity low diversity
# method described in the paper.

export EXPERIMENT_NAME=imagenet64_baseline_ctd
export OUTPUT_DIR=/scratch0/ilya/locDoc/tfgan/examples/${EXPERIMENT_NAME}
export DATA_DIR=/scratch1/ilya/locDoc/data/tfdf

export BATCH_SIZE=128
export DATASET_ARGS='--image_size=64 --dataset_name=imagenet_resized/64x64 --num_classes=1000 --dataset_val_split_name=validation'

export ADDITIONAL='--critic_type=acgan_multiproj \
--num_eval_steps=390 \
--mode=continuous_eval \
--extra_eval_metrics'

bash gpu/_eval_base.sh

# results:

### normal:
# Inference Time : 471.37695s
# Saving dict for global step 999999: discriminator_loss = 1.5117112
# eval/discriminator_val_acc = 0.00038060897435897436
# eval/fid = 15.482412
# eval/generator_inception_acc = 0.2134014423076923
# eval/generator_self_acc = 0.00014022435897435896
# eval/incscore = 19.486443
# eval/real_incscore = 56.66643
# eval/val_real = 0.5971955128205129
# generator_loss = 0.58924365
# global_step = 999999
# loss = 1.5117112

### high fidelity low diversity:
# Inference Time : 451.05388s
# Saving dict for global step 1014999: discriminator_loss = 1.7416052
# eval/discriminator_val_acc = 0.2360977564102564
# eval/fid = 10.249778
# eval/generator_inception_acc = 0.3592948717948718
# eval/generator_self_acc = 0.6275440705128205
# eval/incscore = 29.656748
# eval/real_incscore = 56.666527
# eval/val_real = 0.5851161858974359
# generator_loss = 0.17687708
# global_step = 1014999
# loss = 1.7416052
