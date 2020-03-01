SPLIT=test
GOLD_PATH=/home/hlu/unilm/cnndm.eval/${SPLIT}.target
# ${MODEL_PATH}.${SPLIT} is the predicted target file
python evaluations/eval_for_cnndm.py --pred /home/hlu/notebooks/NLP/examples/text_summarization/nlp_cnndm_finetuning_results_test.txt --gold ${GOLD_PATH} --split ${SPLIT}
