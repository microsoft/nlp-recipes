# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This script reuses some code from https://github.com/nlpyang/Presumm
# This script reuses some code from https://github.com/huggingface/transformers/
# Add to noticefile

from collections import namedtuple
import functools
import logging
from multiprocessing import Pool, cpu_count
import os
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import (
    DataLoader,
    SequentialSampler,
    RandomSampler,
)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from torch.utils.data.distributed import DistributedSampler
from transformers import BertModel

from utils_nlp.common.pytorch_utils import (
    compute_training_steps,
    get_device,
    get_amp,
    move_model_to_device,
    parallelize_model,
)
from utils_nlp.eval import compute_rouge_python
from utils_nlp.models.transformers.common import Transformer
# from utils_nlp.models.transformers.common import TOKENIZER_CLASS




#from transformers.modeling_bart import BART_PRETRAINED_MODEL_ARCHIVE_MAP

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelWithLMHead,
    AutoTokenizer,
)

MODEL_MODES = {
    "language-modeling": AutoModelWithLMHead,
}

from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_CLASS = {
    "bart-large-cnn": BartForConditionalGeneration,
    "t5-large":T5ForConditionalGeneration
}
TOKENIZER_CLASS = {
    "bart-large-cnn": BartTokenizer,
    "t5-large":  T5Tokenizer
}

logger = logging.getLogger(__name__)


import os
import torch
from torch.utils.data import Dataset

from transformers.tokenization_utils import trim_batch

def encode_example(example, tokenizer=None, prefix="", max_source_length=None, max_target_length=None, pad_to_max_length=True, return_tensors="pt"):
    ## add to the dataset
    tokenized_source = tokenizer.batch_encode_plus(
        [prefix + example['src']], max_length=max_source_length, pad_to_max_length=pad_to_max_length, return_tensors=return_tensors,
    )

    source_ids = tokenized_source["input_ids"].squeeze()
    src_mask = tokenized_source["attention_mask"].squeeze()
    example["source_ids"] = source_ids
    example["source_mask"] = src_mask
    if 'tgt' in example: 
        tokenized_target = tokenizer.batch_encode_plus(
        [example['tgt']], max_length=max_target_length, pad_to_max_length=pad_to_max_length, return_tensors=return_tensors,
    )
        target_ids = tokenized_target["input_ids"].squeeze()
        example["target_ids"] = target_ids
    return example

def parallel_preprocess(input_data, preprocess, num_pool=-1):
    """
    Process data in parallel using multiple GPUs.

    Args:
        input_data (list): List if input strings to process.
        preprocess (function): function to apply on the input data.
        word_tokenize (func, optional): A tokenization function used to tokenize
            the results from preprocess_pipeline.
        num_pool (int, optional): Number of CPUs to use. Defaults to -1 and all
            available CPUs are used.

    Returns:
        list: list of processed text strings.

    """
    if num_pool == -1:
        num_pool = cpu_count()

    num_pool = min(num_pool, len(input_data))

    result = None
    with Pool(num_pool) as p:
        results = p.map(
        preprocess, input_data, chunksize=max(1, int(len(input_data) / num_pool)),
        )
    
    p.close()
    #p.join()

    return results


class SummarizationProcessor:
    def __init__(
        self,
        model_name,
        cache_dir="./",
        max_source_length=1024,
        max_target_length=56,
    ):
        #super().__init__()
        self.tokenizer = TOKENIZER_CLASS[model_name].from_pretrained(model_name, cache_dir=cache_dir) # b
        config = AutoConfig.from_pretrained(
            model_name,
            #self.hparams.config_name if self.hparams.config_name else self.hparams.model_name_or_path,
            #**({"num_labels": num_labels} if num_labels is not None else {}),
            cache_dir=cache_dir,
            #**config_kwargs,
        )
        if model_name.startswith("t5"):
        # update config with summarization specific params
            task_specific_params = config.task_specific_params
            if task_specific_params is not None:
                config.update(task_specific_params.get("summarization", {}))

        self.prefix = config.prefix
        #self.source = source_examples #encode_file(tokenizer, os.path.join(data_dir, type_path + ".source"), max_source_length)
        self.with_target = False
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        #if with_target:
        #    self.with_target = True
        #    self.target = source_examples #encode_file(tokenizer, os.path.join(data_dir, type_path + ".target"), max_target_length)

    def preprocess(self, input_data_list):
        preprocess = functools.partial(
            encode_example, tokenizer=self.tokenizer, prefix=self.prefix,  max_source_length=self.max_source_length, max_target_length=self.max_target_length
        )
        
        return parallel_preprocess(input_data_list, preprocess, num_pool=-1)

    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id):
        y = trim_batch(batch["target_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(batch["source_ids"], pad_token_id, attention_mask=batch["source_mask"])
        return source_ids, source_mask, y

    def collate_fn(self, batch, device, train_mode=False):
        input_ids = torch.stack([x["source_ids"] for x in batch])
        masks = torch.stack([x["source_mask"] for x in batch])
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        if train_mode:
            target_ids = torch.stack([x["target_ids"] for x in batch])
            y = trim_batch(target_ids, pad_token_id)
            return {"source_ids": source_ids.to(device), "source_mask": source_mask.to(device), "target_ids": y.to(device)}
        else:
            return {"source_ids": source_ids.to(device), "source_mask": source_mask.to(device)}


class AbstractiveSummarizer(Transformer):
    """class which performs abstractive summarization fine tuning and
        prediction based on BertSumAbs model  """

    def __init__(
        self,
        processor,
        model_name="bart-large-cnn",
        cache_dir=".",
        max_source_length=1024,
        max_target_length=240
    ):
        """Initialize an object of BertSumAbs.

        Args:
            model_name (str, optional:) Name of the pretrained model which is used
                to initialize the encoder of the BertSumAbs model.
                check MODEL_CLASS for supported models. Defaults to "bert-base-uncased".
            cache_dir (str, optional): Directory to cache the tokenizer. Defaults to ".".
            max_pos_length (int, optional): maximum postional embedding length for the
                input. Defaults to 768.
        """

        """super().__init__(
            model_class=AutoModelWithLMHead,
            model_name=model_name,
            num_labels=0,
            cache_dir=cache_dir,
        )
        """
        """
        if model_name not in self.list_supported_models():
            raise ValueError(
                "Model name {} is not supported by BertSumAbs. "
                "Call 'BertSumAbs.list_supported_models()' to get all supported model "
                "names.".format(value)
            )
        """
        self.processor = processor
        self.config = AutoConfig.from_pretrained(
            model_name,
            #self.hparams.config_name if self.hparams.config_name else self.hparams.model_name_or_path,
            #**({"num_labels": num_labels} if num_labels is not None else {}),
            cache_dir=cache_dir,
            #**config_kwargs,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            #self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
            cache_dir=cache_dir,
        )

        self._model_name = model_name
        self.model = MODEL_MODES["language-modeling"].from_pretrained(
            self.model_name,
            #from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
            config=self.config,
            cache_dir=cache_dir,
        )

        self.model_class = AutoModelWithLMHead #MODEL_CLASS[model_name]
        self.cache_dir = cache_dir
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length


    @staticmethod
    def list_supported_models():
        return list(MODEL_CLASS)

    def fit(
        self,
        train_dataset,
        num_gpus=None,
        gpu_ids=None,
        batch_size=4,
        local_rank=-1,
        max_steps=5e4,
        warmup_steps=20000,
        learning_rate=0.002,
        weight_decay=0.01,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,
        report_every=10,
        save_every=1000,
        verbose=True,
        seed=None,
        fp16=False,
        fp16_opt_level="O2",
        world_size=1,
        rank=0,
        validation_function=None,
        checkpoint=None,
        **kwargs,
    ):
        """
        Fine-tune pre-trained transofmer models for extractive summarization.

        Args:
            train_dataset (SummarizationDataset): Training dataset.
            num_gpus (int, optional): The number of GPUs to use. If None, all
                available GPUs will be used. If set to 0 or GPUs are not available,
                CPU device will be used. Defaults to None.
            gpu_ids (list): List of GPU IDs to be used.
                If set to None, the first num_gpus GPUs will be used.
                Defaults to None.
            batch_size (int, optional): Maximum number of tokens in each batch.
            local_rank (int, optional): Local_rank for distributed training on GPUs.
                Local rank means the ranking of the current GPU device on the current
                node. Defaults to -1, which means non-distributed training.
            max_steps (int, optional): Maximum number of training steps. Defaults to 5e5.
            warmup_steps_bert (int, optional): Number of steps taken to increase
                learning rate from 0 to `learning_rate` for tuning the BERT encoder.
                Defaults to 2e4.
            warmup_steps_dec (int, optional): Number of steps taken to increase
                learning rate from 0 to `learning_rate` for tuning the decoder.
                Defaults to 1e4.
            learning_rate_bert (float, optional):  Learning rate of the optimizer
                for the encoder. Defaults to 0.002.
            learning_rate_dec (float, optional):  Learning rate of the optimizer
                for the decoder. Defaults to 0.2.
            optimization_method (string, optional): Optimization method used in fine
                tuning. Defaults to "adam".
            max_grad_norm (float, optional): Maximum gradient norm for gradient clipping.
                Defaults to 0.
            beta1 (float, optional): The exponential decay rate for the first moment
                estimates. Defaults to 0.9.
            beta2 (float, optional): The exponential decay rate for the second-moment
                estimates. This value should be set close to 1.0 on problems with
                a sparse gradient. Defaults to 0.99.
            decay_method (string, optional): learning rate decrease method.
                Default to 'noam'.
            gradient_accumulation_steps (int, optional): Number of batches to accumulate
                gradients on between each model parameter update. Defaults to 1.
            report_every (int, optional): The interval by steps to print out the
                training log. Defaults to 10.
            save_every (int, optional): The interval by steps to save the finetuned 
                model. Defaults to 100.
            verbose (bool, optional): Whether to print out the training log.
                Defaults to True.
            seed (int, optional): Random seed used to improve reproducibility.
                Defaults to None.
            fp16 (bool, optional): Whether to use mixed precision training.
                Defaults to False.
            fp16_opt_level (str, optional): optimization level, refer to
                 https://nvidia.github.io/apex/amp.html#opt-levels for details.
                 Value choices are: "O0", "O1", "O2", "O3". Defaults to "O2".
            world_size (int, optional): Total number of GPUs that will be used.
                Defaults to 1.
            rank (int, optional): Global rank of the current GPU in distributed
                training. It's calculated with the rank of the current node in the
                cluster/world and the `local_rank` of the device in the current node.
                See an example in :file: `examples/text_summarization/
                abstractive_summarization_bertsum_cnndm_distributed_train.py`.
                Defaults to 0.
            validation_function (function, optional): function used in fitting to
                validate the performance. Default to None.
            checkpoint (str, optional): file path for a checkpoint based on which the
                training continues. Default to None.
        """

        # move model to devices
        print("device is {}".format(device))
        if checkpoint:
            # checkpoint should have "model", "optimizer", "amp"
            checkpoint_state_dict = torch.load(checkpoint, map_location="cpu") 
            
        # init optimizer
        device, num_gpus, amp = self.prepare_model_and_optimizer(
            num_gpus=num_gpus,
            gpu_ids=gpu_ids,
            local_rank=local_rank,
            fp16=fp16,
            fp16_opt_level=fp16_opt_level,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            adam_epsilon=adam_epsilon,
            checkpoint_state_dict=checkpoint_state_dict,
        )
       
        global_step = 0
        if "global_step" in checkpoint_state_dict and checkpoint_state_dict["global_step"]:
            global_step = checkpoint_state_dict["global_step"] / world_size
            print("global_step is {}".format(global_step))

        self.scheduler = Transformer.get_default_scheduler(
            optimizer=self.optimizer,
            warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
        if global_step > 0:
            self.scheduler.load_state_dict(checkpoint_state_dict["lr_scheduler"])

        if local_rank == -1:
            sampler = RandomSampler(train_dataset)
        else:
            sampler = DistributedSampler(
                train_dataset, num_replicas=world_size, rank=rank
            )

        def collate_fn(data):
            return self.processor.collate(
                data, block_size=self.max_pos_length, device=device
            )

        train_dataloader = DataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

        # compute the max number of training steps
        max_steps = compute_training_steps(
            train_dataloader,
            max_steps=max_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        super().fine_tune(
            train_dataloader=train_dataloader,
            get_inputs=xxxx.get_inputs,
            device=device,
            num_gpus=num_gpus,
            max_steps=max_steps,
            global_step=global_step,
            max_grad_norm=max_grad_norm,
            gradient_accumulation_steps=gradient_accumulation_steps,
            verbose=verbose,
            seed=seed,
            report_every=report_every,
            save_every=save_every,
            clip_grad_norm=False,
            optimizer=optimizers,
            scheduler=None,
            fp16=fp16,
            amp=self.amp,
            validation_function=validation_function,
        )

        # release GPU memories
        self.model.cpu()
        torch.cuda.empty_cache()

        self.save_model(max_steps)

    def predict(
        self,
        test_dataset,
        num_gpus=None,
        gpu_ids=None,
        local_rank=-1,
        batch_size=16,
        alpha=0.6,
        beam_size=5,
        min_length=15,
        max_length=150,
        fp16=False,
        verbose=True,
    ):
        """
        Predict the summarization for the input data iterator.

        Args:
            test_dataset (SummarizationDataset): Dataset for which the summary
                to be predicted.
            num_gpus (int, optional): The number of GPUs used in prediction.
                Defaults to 1.
            gpu_ids (list): List of GPU IDs to be used.
                If set to None, the first num_gpus GPUs will be used.
                Defaults to None.
            local_rank (int, optional): Local rank of the device in distributed
                inferencing. Defaults to -1, which means non-distributed inferencing.
            batch_size (int, optional): The number of test examples in each batch.
                Defaults to 16.
            alpha (float, optional): Length penalty. Defaults to 0.6.
            beam_size (int, optional): Beam size of beam search. Defaults to 5.
            min_length (int, optional): Minimum number of tokens in the output sequence.
                Defaults to 15.
            max_length (int, optional):  Maximum number of tokens in output
                sequence. Defaults to 150.
            fp16 (bool, optional): Whether to use half-precision model for prediction.
                Defaults to False.
            verbose (bool, optional): Whether to print out the training log.
                Defaults to True.

        Returns:
            List of strings which are the summaries

        """
        device, num_gpus = get_device(
            num_gpus=num_gpus, gpu_ids=gpu_ids, local_rank=local_rank
        )

        if fp16:
            self.model = self.model.half()

        self.model = move_model_to_device(self.model, device)
        self.model.eval()

        self.model = parallelize_model(
            self.model,
            device,
            num_gpus=num_gpus,
            gpu_ids=gpu_ids,
            local_rank=local_rank,
        )

        test_sampler = SequentialSampler(test_dataset)

        def collate_fn(data):
            return self.processor.collate_fn(
                data, device, train_mode=False
            )

        test_dataloader = DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )
        print("dataset length is {}".format(len(test_dataset)))
        generated_summaries = []

        for batch in tqdm(
            test_dataloader, desc="Generating summary", disable=not verbose
        ):
            #if self.model_name.startswith("t5"):
            #    batch = [self.model.config.prefix + text for text in batch]
            #dct = self.tokenizer.batch_encode_plus(batch, max_length=1024, return_tensors="pt", pad_to_max_length=True)
            print(batch)
            summaries = self.model.module.generate(
                input_ids=batch["source_ids"],
                attention_mask=batch["source_mask"],
                min_length=min_length,
                max_length=max_length
            )
            decoded_summaries = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
            generated_summaries.extend(decoded_summaries)

        # release GPU memories
        self.model.cpu()
        torch.cuda.empty_cache()

        return generated_summaries

    def save_model(self, global_step=None, full_name=None):
        """
        save the trained model.

        Args:
            global_step (int, optional): The number of steps that the model has been
                finetuned for. Defaults to None.
            full_name (str, optional): File name to save the model's `state_dict()`.
                If it's None, the model is going to be saved under "fine_tuned" folder
                of the cached directory of the object. Defaults to None.
        """
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Take care of distributed/parallel training

        if full_name is None:
            output_model_dir = os.path.join(self.cache_dir, "fine_tuned")
            os.makedirs(self.cache_dir, exist_ok=True)
            os.makedirs(output_model_dir, exist_ok=True)
            full_name = os.path.join(output_model_dir, "bertsumabs.pt")
        else:
            path, filename = os.path.split(full_name)
            print(path)
            os.makedirs(path, exist_ok=True)

        checkpoint = {
            "optimizers": [self.optim_bert.state_dict(), self.optim_dec.state_dict()],
            "model": model_to_save.state_dict(),
            "amp": self.amp.state_dict() if self.amp else None,
            "global_step": global_step,
            "max_pos_length": self.max_pos_length,
        }

        logger.info("Saving model checkpoint to %s", full_name)
        try:
            print("saving through pytorch to {}".format(full_name))
            torch.save(checkpoint, full_name)
        except OSError:
            try:
                print("saving as pickle")
                pickle.dump(checkpoint, open(full_name, "wb"))
            except Exception:
                raise
        except Exception:
            raise
