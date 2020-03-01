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
