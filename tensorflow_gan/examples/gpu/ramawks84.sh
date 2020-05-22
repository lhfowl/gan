export DATA_DIR=/scratch2/ilyak/locDoc/data/tfdf
export OUTPUT_DIR=/scratch2/ilyak/locDoc/tfgan/logdir

export ADDITIONAL='--aux_cond_generator_weight=0.1 --aux_cond_discriminator_weight=1.0'

bash gpu/_base.sh