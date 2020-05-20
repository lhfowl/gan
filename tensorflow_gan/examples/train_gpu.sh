python self_attention_estimator/train_experiment_main.py \
  --use_tpu=false \
  --eval_on_tpu=false \
  --use_tpu_estimator=false \
  --mode=train_and_eval \
  --max_number_of_steps=999999 \
  --train_batch_size=16 \
  --eval_batch_size=16 \
  --predict_batch_size=16 \
  --num_eval_steps=10 \
  --train_steps_per_eval=1000 \
  --model_dir=/scratch0/ilya/locDoc/tfgan/logdir \
  --imagenet_data_dir=/scratch0/ilya/locDoc/data/tfdf \
  --alsologtostderr