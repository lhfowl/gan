export DATA_DIR=/scratch0/ilya/locDoc/data/tfdf
export OUTPUT_DIR=/scratch0/ilya/locDoc/tfgan/logdir

export BATCH_SIZE=64
export ADDITIONAL='--critic_type=kplusone_fm --kplusone_mhinge_cond_discriminator_weight=1.0 --aux_mhinge_cond_generator_weight=0.05'

bash gpu/_base.sh