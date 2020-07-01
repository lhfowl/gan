

python3.7 self_attention_estimator/train_experiment_main.py \
  --use_tpu=true \
  --eval_on_tpu=true \
  --use_tpu_estimator=true \
  --mode=train_and_eval \
  --max_number_of_steps=9999999 \
  --train_batch_size=${BATCH_SIZE} \
  --eval_batch_size=${EVAL_BATCH_SIZE} \
  --predict_batch_size=${BATCH_SIZE} \
  --num_eval_steps=10 \
  --train_steps_per_eval=${TRAIN_STEPS_PER_EVAL} \
  --tpu=$TPU_NAME \
  --gcp_project=$PROJECT_ID \
  --tpu_zone=$TPU_ZONE \
  --model_dir=${STORAGE_BUCKET}/experiments/${EXPERIMENT_NAME} \
  --data_dir=$STORAGE_BUCKET/data \
  ${DATASET_ARGS} \
  --alsologtostderr ${ADDITIONAL}