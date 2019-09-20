# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script reuses some code from
# https://github.com/huggingface/pytorch-transformers/blob/master/examples/run_glue.py


from pytorch_transformers.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP
from pytorch_transformers.modeling_xlnet import XLNET_PRETRAINED_MODEL_ARCHIVE_MAP
from pytorch_transformers.modeling_roberta import ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
from pytorch_transformers.modeling_distilbert import DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP
from pytorch_transformers import (
    BertTokenizer,
    DistilBertTokenizer,
    RobertaTokenizer,
    XLNetTokenizer,
)

TOKENIZER_CLASS = {}
TOKENIZER_CLASS.update({k: BertTokenizer for k in BERT_PRETRAINED_MODEL_ARCHIVE_MAP})
TOKENIZER_CLASS.update({k: RobertaTokenizer for k in ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP})
TOKENIZER_CLASS.update({k: XLNetTokenizer for k in XLNET_PRETRAINED_MODEL_ARCHIVE_MAP})
TOKENIZER_CLASS.update({k: DistilBertTokenizer for k in DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP})

MAX_SEQ_LEN = 512


def fine_tune(
    model,
    model_type,
    train_dataloader,
    num_epochs,
    device,
    n_gpu,
    local_rank,
    gradient_accumulation_steps,
    fp16,
    seed,
):
    global_step = 0
    tr_loss = 0.0
    model.zero_grad()
    train_iterator = trange(int(num_epochs), desc="Epoch", disable=local_rank not in [-1, 0])
    set_seed(seed)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2] if model_type in ["bert", "xlnet"] else None,
                "labels": batch[3],
            }
            outputs = model(**inputs)
            loss = outputs[0]

            if n_gpu > 1:
                loss = loss.mean()
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                scheduler.step()
                optimizer.step()
                model.zero_grad()
                global_step += 1

            if max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if max_steps > 0 and global_step > max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step
