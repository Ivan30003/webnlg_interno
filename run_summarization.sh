export TOKENIZERS_PARALLELISM="false"

OUTPUT_DIR="experiments/FRED/full_translated"
MODEL_PATH="models/FRED_T5_1.7B"
TRAIN_FILE="data/created_datasets/full/translated/ru_train_dataset_ru.jsonl"
VAL_FILE="data/created_datasets/full/translated/ru_dev_dataset_ru.jsonl"
TEST_FILE="data/created_datasets/full/translated/ru_test_dataset_ru.jsonl"

mkdir -p $OUTPUT_DIR
cp $0 $OUTPUT_DIR

torchrun --nproc_per_node 4 --master_port 4207 run_summarization.py \
    --output_dir $OUTPUT_DIR/model \
    --model_name_or_path $MODEL_PATH \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file $TRAIN_FILE \
    --validation_file $VAL_FILE \
    --test_file $TEST_FILE \
    --source_prefix "<LM>" \
    --label_postfix "</s>" \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --predict_with_generate \
    --text_column "prompt" \
    --summary_column "completion" \
    --evaluation_strategy "steps" \
    --eval_steps 0.025 \
    --save_steps 0.025 \
    --logging_dir $OUTPUT_DIR/log \
    --logging_strategy "steps" \
    --logging_steps 0.01 \
    --save_total_limit 2 \
    --report_to "tensorboard" \
    --label_smoothing_factor 0.1 \
    --warmup_ratio 0.05 \
    --num_train_epochs 20 \
    --learning_rate 5e-4 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type "cosine" \
    --is_peft_model True \
    --ddp_find_unused_parameters False \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_meteor" \
    --meteor_path "metrics/webnlg_2023/evaluation/automatic/scripts/metrics/meteor-1.5/meteor-1.5.jar"
