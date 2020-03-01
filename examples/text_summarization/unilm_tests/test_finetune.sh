TRAIN_FILE=/home/hlu/unilm/cnndm_json/train_file_cnndm_full.json
CACHED_FEATURE_FILE=/home/hlu/unilm/cnndm_finetuned_s2s/cnndm_train.cased.features.pt
# path where fine-tuned checkpoints are saved
OUTPUT_DIR=/home/hlu/unilm/cnndm_finetuned_s2s
CACHE_DIR=/home/hlu/unilm/cnndm_finetuned_s2s
# path of pre-trained model
PRETRAIN_PATH=/home/hlu/unilm/unilmv1-large-cased.bin

export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m torch.distributed.launch --nproc_per_node=4 run_seq2seq.py \
  --train_file ${TRAIN_FILE} --output_dir ${OUTPUT_DIR} --model_type unilm --model_name_or_path unilm-large-cased \
  --cached_train_features_file $CACHED_FEATURE_FILE --fp16 --fp16_opt_level O2 \
  --max_source_seq_length 640 --max_target_seq_length 128 --per_gpu_train_batch_size 2 --gradient_accumulation_steps 2 \
  --learning_rate 3e-5 --num_warmup_steps 500 --num_training_steps 5000 --cache_dir ${CACHE_DIR} --save_steps 5000
