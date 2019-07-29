# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script reuses some code from
# https://github.com/huggingface/pytorch-transformers/blob/067923d3267325f525f4e46f357360c191ba562e/examples/run_squad.py
# https://github.com/huggingface/pytorch-transformers/blob/067923d3267325f525f4e46f357360c191ba562e/examples/utils_squad.py
# https://github.com/huggingface/pytorch-transformers/blob/067923d3267325f525f4e46f357360c191ba562e/examples/utils_squad_evaluate.py


import os
import logging
from tqdm import tqdm, trange

import torch
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)

from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers.modeling_bert import BertConfig, BertForQuestionAnswering

from utils_nlp.models.bert.common import Language
from utils_nlp.common.pytorch_utils import get_device, move_to_device


logger = logging.getLogger(__name__)

class BERTQAExtractor:
    def __init__(self, language=Language.ENGLISH, cache_dir=".", fine_tuned_model=None):
                
        self.language = language
        self.cache_dir = cache_dir


        if fine_tuned_model:
            self.model = BertForQuestionAnswering.from_pretrained(fine_tuned_model)
        else:
            config = BertConfig.from_pretrained(language.value)
            self.model = BertForQuestionAnswering.from_pretrained(
                language.value, config=config
            )

    def fit(
        self, 
        features,
        num_gpus=None,
        num_epochs=1,
        batch_size=32,
        lr=2e-5,
        warmup_proportion=None,
        max_grad_norm=1.0,
        model_output_dir=None
        ):
        device = get_device("cpu" if num_gpus == 0 or not torch.cuda.is_available() else "gpu")
        self.model = move_to_device(self.model, device, num_gpus)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        train_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                      all_start_positions, all_end_positions)

        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

        t_total = len(train_dataloader) * num_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)

        if warmup_proportion:
            warmup_steps = t_total * warmup_proportion
        else:
            warmup_steps = 0

        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        train_iterator = trange(int(num_epochs), desc="Epoch")
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(device) for t in batch)
                inputs = {'input_ids':       batch[0],
                          'attention_mask':  batch[1], 
                          'token_type_ids':  batch[2],  
                          'start_positions': batch[3], 
                          'end_positions':   batch[4]}

                outputs = self.model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers

                loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training


                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                tr_loss += loss.item()

                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                self.model.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/50, global_step)
                    logging_loss = tr_loss

            epoch_iterator.close()
        train_iterator.close()

        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        print ("Average training loss: {}".format(tr_loss / global_step))

        if model_output_dir:
            if not os.path.exists(model_output_dir):
                os.makedirs(model_output_dir)

            logger.info("Saving model checkpoint to %s", model_output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = self.model.module if hasattr(self.model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(model_output_dir)


    def predict(
        self,
        features,
        num_gpus=None,
        batch_size=32,
        probabilities=False):
        
        device = get_device("cpu" if num_gpus == 0 or not torch.cuda.is_available() else "gpu")
        self.model = move_to_device(self.model, device, num_gpus)

        # score
        self.model.eval()

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        test_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                     all_example_index)

        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

        all_results = []
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2]
                        }
                example_indices = batch[3]

                outputs = self.model(**inputs)

            for i, example_index in enumerate(example_indices):
                test_feature = features[example_index.item()]
                unique_id = int(test_feature.unique_id)

                result = RawResult(unique_id    = unique_id,
                                   start_logits = to_list(outputs[0][i]),
                                   end_logits   = to_list(outputs[1][i]))
                all_results.append(result)
