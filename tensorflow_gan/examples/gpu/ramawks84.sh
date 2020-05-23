export DATA_DIR=/scratch2/ilyak/locDoc/data/tfdf
export OUTPUT_DIR=/scratch2/ilyak/locDoc/tfgan/logdir

# export ADDITIONAL='--aux_cond_generator_weight=0.1 --aux_cond_discriminator_weight=1.0'

export BATCH_SIZE=64
export ADDITIONAL='--critic_type=kplusone_wgan --kplusone_mhinge_cond_discriminator_weight=1.0 --aux_mhinge_cond_generator_weight=1.0'


bash gpu/_base.sh