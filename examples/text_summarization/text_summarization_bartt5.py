from pathlib import Path
from transformers import BartForConditionalGeneration, BartTokenizer
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer


model_class = {
    "bart-large-cnn": BartForConditionalGeneration,
    "t5-large":T5ForConditionalGeneration
}
tokenizer_class = {
    "bart-large-cnn": BartTokenizer,
    "t5-large":  T5Tokenizer
}
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def generate_summaries(
    examples: list, out_file: str, model_name: str, batch_size: int = 8, device: str = "cuda"
):
    fout = Path(out_file).open("w")
    model = model_class[model_name].from_pretrained(model_name).to(device)
    tokenizer = tokenizer_class[model_name].from_pretrained(model_name) # bart-large

    max_length = 140
    min_length = 55

    if model_name.startswith("t5"):
        # update config with summarization specific params
        task_specific_params = model.config.task_specific_params
        if task_specific_params is not None:
            model.config.update(task_specific_params.get("summarization", {}))

    for batch in tqdm(list(chunks(examples, batch_size))):
        if model_name.startswith("t5"):
            batch = [model.config.prefix + text for text in batch]
        dct = tokenizer.batch_encode_plus(batch, max_length=1024, return_tensors="pt", pad_to_max_length=True)
        summaries = model.generate(
            input_ids=dct["input_ids"].to(device),
            attention_mask=dct["attention_mask"].to(device),
            #num_beams=4,
            #length_penalty=2.0,
            #max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
            #min_length=min_length + 1,  # +1 from original because we start at step=1
            #no_repeat_ngram_size=3,
            #early_stopping=True,
            #decoder_start_token_id=model.config.eos_token_id,
        )
        dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
        for hypothesis in dec:
            fout.write(hypothesis + "\n")
            fout.flush()

examples = [" " + x.rstrip() for x in open("./cnn_test.txt").readlines()]
generate_summaries(examples, "./cnn_generated.txt", "bart-large-cnn", batch_size=4, device="cuda")
#generate_summaries(examples, "./cnn_generated-t5.txt", "t5-large", batch_size=4, device="cuda")