### Inference validation, 
## run from folder unilm/s2s-ft
## Infer using cnndm_model.bin on cnndm.eval/test.target, about 1.5 hours
# ROUGE-1: 0.4275295152882667     ROUGE-2: 0.2020790151284723
## test_decode.sh
# path of fine-tuned checkpoint
MODEL_PATH=/home/hlu/unilm/cnndm_model.bin
SPLIT=test
# input file that you would like to decode
INPUT_JSON=/home/hlu/unilm/cnndm_json/test_file_cnndm_full.json

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python decode_seq2seq.py \
  --fp16 --model_name unilm1-large-cased  --input_file ${INPUT_JSON} --split $SPLIT \
  --model_recover_path ${MODEL_PATH} --max_seq_length 768 --max_tgt_length 128 --batch_size 96 --beam_size 5 \
  --length_penalty 0 --forbid_duplicate_ngrams --mode s2s --forbid_ignore_word "."
  
#evaluate.sh
SPLIT=test
GOLD_PATH=/home/hlu/unilm/cnndm.eval/${SPLIT}.target
# ${MODEL_PATH}.${SPLIT} is the predicted target file
python evaluations/eval_for_cnndm.py --pred /home/hlu/unilm/cnndm_model.bin.test --gold ${GOLD_PATH} --split ${SPLIT}

## Infer with nlp predict method
## run from abstractive_summarization_unilm_cnndm.ipynb
# ROUGE-1: 0.4274837359516832     ROUGE-2: 0.2021257397922513
## load model
MODEL_NAME = "unilm-large-cased"
abs_summarizer = S2SAbstractiveSummarizer(
    model_name=MODEL_NAME,
    load_model_from_dir="/home/hlu/unilm/",
    model_file_name="cnndm_model.bin",
    max_seq_len=768,
    max_source_seq_length=660,
    max_target_seq_length=128,
)
## predict on fine-tuned model checkpoint
res = abs_summarizer.predict(
   test_dataset=test_dataset,
   per_gpu_batch_size=24,
   fp16=True,
   beam_size=5,
   forbid_ignore_word=".")

### Fine tuning validation
## Fine tuning using s2s-ft for 5000 steps, about 1.5 hours
# ROUGE-1: 0.38867918232138565    ROUGE-2: 0.17830446095107455
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

## predict on the 5000 steps s2s model
# test_decode.sh
# path of fine-tuned checkpoint
# MODEL_PATH=/home/hlu/unilm/cnndm_model.bin
MODEL_PATH=/home/hlu/unilm/cnndm_finetuned_s2s/model.5000.bin
SPLIT=test
# input file that you would like to decode
INPUT_JSON=/home/hlu/unilm/cnndm_json/test_file_cnndm_full.json

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python decode_seq2seq.py \
  --fp16 --model_name unilm-large-cased  --input_file ${INPUT_JSON} --split $SPLIT \
  --model_recover_path ${MODEL_PATH} --max_seq_length 768 --max_tgt_length 128 --batch_size 96 --beam_size 5 \
  --length_penalty 0 --forbid_duplicate_ngrams --mode s2s --forbid_ignore_word "."


## Fine tuning with nlp fit method, DP, no FP16, per gpu batch size 1
## run from abstractive_summarization_unilm_cnndm.ipynb
# 02/20/2020 18:07:53 - INFO - __main__ -   ***** Evaluation: /home/hlu/notebooks/NLP/examples/text_summarization/nlp_cnndm_finetuning_results.txt *****
# /home/hlu/notebooks/NLP/examples/text_summarization/nlp_cnndm_finetuning_results.txt
# ROUGE-1: 0.3782984504336976     ROUGE-2: 0.17248707008580857
## load model
abs_summarizer = S2SAbstractiveSummarizer(
    model_name=MODEL_NAME,
    max_seq_len=768,
    max_source_seq_length=640,
    max_target_seq_length=128,
)

## fit model
abs_summarizer.fit(
    train_dataset=train_dataset,
    per_gpu_batch_size=PER_GPU_BATCH_SIZE,
    gradient_accumulation_steps=2,
    learning_rate=3e-5,
    warmup_steps=500,
    max_steps=5000,
)

## predict on fine-tuned model
res = abs_summarizer.predict(
   test_dataset=test_dataset,
   per_gpu_batch_size=24,
   fp16=True,
   beam_size=5,
   forbid_ignore_word=".")


## DDP, FP16, O2, per gpu batch size 2
# 02/23/2020 20:31:53 - INFO - __main__ -   ***** Evaluation: /home/hlu/notebooks/NLP/examples/text_summarization/nlp_cnndm_finetuning_results_ddp.txt *****
# /home/hlu/notebooks/NLP/examples/text_summarization/nlp_cnndm_finetuning_results_ddp.txt
# ROUGE-1: 0.39192990436578873    ROUGE-2: 0.17944043767459286
# run abstractive_summarization_unilm_cnndm.py, set QUICK_RUN to FALSE
python abstractive_summarization_unilm_cnndm.py -fp16 true

