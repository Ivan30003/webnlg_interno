export TOKENIZERS_PARALLELISM="false"

OUTPUT_DIR="submission"
MODEL_PATH="models/FRED_T5_1.7B"
PEFT_MODEL_PATH=$OUTPUT_DIR/model/checkpoint-10968
TEST_FILE="data/created_datasets/full/translated/ru_test_dataset_ru_inference.jsonl"

torchrun --nproc_per_node 2 run_summarization.py \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_PATH \
    --peft_model_id $PEFT_MODEL_PATH \
    --do_predict \
    --test_file $TEST_FILE \
    --source_prefix "<LM>" \
    --label_postfix "</s>" \
    --overwrite_output_dir \
    --per_device_eval_batch_size 6 \
    --predict_with_generate \
    --text_column "prompt" \
    --summary_column "completion" \
    --report_to "tensorboard" \
    --is_peft_model True \
    --num_beams 5 \
    --meteor_path "metrics/webnlg_2023/evaluation/automatic/scripts/metrics/meteor-1.5/meteor-1.5.jar"
