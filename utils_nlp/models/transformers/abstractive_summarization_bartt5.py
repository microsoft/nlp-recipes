# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This script reuses some code from https://github.com/huggingface/transformers/


import functools
import logging
import os
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import (
    DataLoader,
    SequentialSampler,
    RandomSampler,
)
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing
from torch import nn

torch.multiprocessing.set_sharing_strategy("file_system")

from transformers import (
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    BartForConditionalGeneration,
    BART_PRETRAINED_MODEL_ARCHIVE_MAP,
    T5ForConditionalGeneration,
    T5_PRETRAINED_MODEL_ARCHIVE_MAP,
)
from transformers.tokenization_utils import trim_batch

from utils_nlp.common.pytorch_utils import (
    compute_training_steps,
    get_device,
    move_model_to_device,
    parallelize_model,
)
from utils_nlp.models.transformers.common import Transformer


MODEL_MODES = {
    "language-modeling": AutoModelWithLMHead,
}

MODEL_CLASS = {}
MODEL_CLASS.update(
    {k: BartForConditionalGeneration for k in BART_PRETRAINED_MODEL_ARCHIVE_MAP}
)
MODEL_CLASS.update(
    {k: T5ForConditionalGeneration for k in T5_PRETRAINED_MODEL_ARCHIVE_MAP}
)

logger = logging.getLogger(__name__)



def encode_example(
    example,
    tokenizer,
    prefix="",
    max_source_length=None,
    max_target_length=None,
    pad_to_max_length=True,
    return_tensors="pt",
):
    """
    Encode a single example with the specified tokenizer.

    Args:
        example:
        tokenizer
        prefix
        max_source_length
        max_target_length:
        pad_to_max_length:
        return_tensors:

    """

    tokenized_source = tokenizer.batch_encode_plus(
        [prefix + example["src"]],
        max_length=max_source_length,
        pad_to_max_length=pad_to_max_length,
        return_tensors=return_tensors,
    )

    source_ids = tokenized_source["input_ids"].squeeze()
    src_mask = tokenized_source["attention_mask"].squeeze()
    example["source_ids"] = source_ids
    example["source_mask"] = src_mask
    if "tgt" in example:
        tokenized_target = tokenizer.batch_encode_plus(
            [example["tgt"]],
            max_length=max_target_length,
            pad_to_max_length=pad_to_max_length,
            return_tensors=return_tensors,
        )
        target_ids = tokenized_target["input_ids"].squeeze()
        example["target_ids"] = target_ids
    return example


class Predictor(nn.Module):
    """
    Predictor which can run on multi-GPUs.

    Args:
        model (AbstractiveSummarizer): the summarizer model which will 
            be used for prediction.
        min_length (int): the minimum generated summary length.
        max_length (int): the maximum generated summary length.
        kwargs (dict): Additional kwargs that will be forwarded
            to `Predictor`. Please consult the arguments in function
            `PreTrainedModel::generate`.

    """

    def __init__(self, model, min_length=55, max_length=140, **kwargs):
        super(Predictor, self).__init__()
        self.model = model.module if hasattr(model, "module") else model
        self.min_length = min_length
        self.max_length = max_length
        self.config = kwargs

    def forward(self, src, src_mask):
        """ Generate sequences for models with a LM head.

        Args: 
            src: (`optional`) `torch.LongTensor` of shape `(batch_size, sequence_length)`
                The sequence used as a prompt for the generation. If `None` the
                method initializes it as an empty `torch.LongTensor` of shape `(1,)`.
            src_mask (`optional`) obj: `torch.LongTensor` of same shape as `input_ids`
                Mask to avoid performing attention on padding token indices.
                Mask values selected in ``[0, 1]``:
                ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
                Defaults to `None`.
        """

        device = src.device
        with torch.no_grad():
            summaries = self.model.generate(
                input_ids=src,
                attention_mask=src_mask,
                min_length=self.min_length,
                max_length=self.max_length,
                **self.config,
            )
            predictions = torch.tensor(
                [
                    i.tolist()[0 : self.max_length]
                    + [0] * (self.max_length - i.size()[0])
                    for i in summaries
                ],
                device=device,
            )

            return predictions


def validate(summarizer, validate_dataset, num_gpus=1, TOP_N=2):
    """ validation function to be used optionally in fine tuning.

    Args:
        summarizer(BertSumAbs): The summarizer under fine tuning.
        validate_dataset (SummarizationDataset): dataset for validation.
        num_gpus (int, optional): number of GPUs used for validation.
            Defaults to 1.
        TOP_N (int, optional): the number of examples used from
            validate_dataset. Defaults to 2.

    Returns:
        None.
    """
    shortened_dataset = validate_dataset[0:TOP_N]
    a = summarizer.processor.collate_fn(shortened_dataset, "cuda:0", True)
    c = summarizer.processor.get_inputs(
        a, "cuda:0", summarizer.model_name, summarizer.tokenizer, True
    )

    output = summarizer.model(**c)
    generated_summaries = summarizer.predict(
        shortened_dataset, num_gpus=num_gpus, batch_size=TOP_N
    )
    print("validation loss is {}".format(output[0]))
    print("prediction is {}".format(generated_summaries[0]))


class SummarizationProcessor:
    """ Class for preprocessing abstractive summarization data for BART/T5 models. 

    Args:
        tokenizer(AutoTokenizer): tokenizer for the model used for preprocessing.
        config(AutoConfig): config for the model used for preprocessing.
        max_source_length (int, optional): Max number of tokens that be used
            as input. Defaults to 1024.
        max_target_length (int, optional): Max number of tokens that be used
                as in target. Defaults to 140.

    """
    def __init__(
        self, tokenizer, config, max_source_length=1024, max_target_length=140,
    ):

        self.tokenizer = tokenizer
        self.config = config

        self.prefix = config.prefix
        self.with_target = False
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id):

        y = trim_batch(batch["target_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(
            batch["source_ids"], pad_token_id, attention_mask=batch["source_mask"]
        )
        return source_ids, source_mask, y

    def preprocess(self, input_data_list):
        result = []
        for i in input_data_list:
            result.append(
                encode_example(
                    i,
                    tokenizer=self.tokenizer,
                    prefix=self.prefix,
                    max_source_length=self.max_source_length,
                    max_target_length=self.max_target_length,
                )
            )
        return result

    @staticmethod
    def get_inputs(batch, device, model_name, tokenizer=None, train_mode=True):
        pad_token_id = tokenizer.pad_token_id
        if not train_mode:
            source_ids, source_mask = batch["source_ids"], batch["source_mask"]
        else:
            source_ids, source_mask, y = SummarizationProcessor.trim_seq2seq_batch(
                batch, pad_token_id
            )
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone()
        lm_labels[y[:, 1:] == pad_token_id] = -100

        if train_mode:
            return {
                "input_ids": source_ids,
                "attention_mask": source_mask,
                "decoder_input_ids": y_ids,
                "lm_labels": lm_labels,
            }
        else:
            return {
                "input_ids": source_ids,
                "attention_mask": source_mask,
            }

    def collate_fn(self, batch, device, train_mode=False):
        input_ids = torch.stack([x["source_ids"] for x in batch])
        masks = torch.stack([x["source_mask"] for x in batch])
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask = trim_batch(
            input_ids, pad_token_id, attention_mask=masks
        )
        if train_mode:
            target_ids = torch.stack([x["target_ids"] for x in batch])
            y = trim_batch(target_ids, pad_token_id)
            return {
                "source_ids": source_ids.to(device),
                "source_mask": source_mask.to(device),
                "target_ids": y.to(device),
            }
        else:
            return {
                "source_ids": source_ids.to(device),
                "source_mask": source_mask.to(device),
            }


class AbstractiveSummarizer(Transformer):
    """class which performs abstractive summarization fine tuning and
        prediction based on BART and T5 model  """

    def __init__(
        self,
        # processor,
        model_name="t5-small",
        cache_dir=".",
        max_source_length=1024,
        max_target_length=240,
    ):
        """Initialize an object of BertSumAbs.

        Args:
            model_name (str, optional:) Name of the pretrained model which is used .
                `AbstractiveSummarizer.list_supported_models()` to see all supported
                model names. Defaults to "t5-small".
            cache_dir (str, optional): Directory to cache the model. Defaults to ".".
            max_source_length (int, optional): maximum source length for the
                input. Defaults to 1024.
            max_target_length (int, optional): maximum target length for the
                training input. Defaults to 240.

        """

        if model_name not in self.list_supported_models():
            raise ValueError(
                "Model name {} is not supported by AbstractiveSummarizer. "
                "Call 'AbstractiveSummarizer.list_supported_models()' to"
                "get all supported model "
                "names.".format(value)
            )

        self.config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir,)
        self.config.output_past = True  # to enable num_beams greater than 1
        task_specific_params = self.config.task_specific_params
        if task_specific_params is not None:
            self.config.update(task_specific_params.get("summarization", {}))
        self.config.update({"max_length": max_target_length})
        self.config.update({"attention_dropout": 0.1})

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir,)

        self.processor = SummarizationProcessor(
            self.tokenizer, self.config, max_source_length, max_target_length
        )

        self._model_name = model_name
        self.model = MODEL_MODES["language-modeling"].from_pretrained(
            self.model_name, config=self.config, cache_dir=cache_dir,
        )

        self.cache_dir = cache_dir
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        self.amp = None
        self.optimizer = None
        self.scheduler = None

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
        warmup_steps=2e3,
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
                available GPUs will be used. If set to 0 or GPUs are
                not available, CPU device will be used. Defaults to None.
            gpu_ids (list): List of GPU IDs to be used.
                If set to None, the first num_gpus GPUs will be used.
                Defaults to None.
            batch_size (int, optional): Maximum number of examples in each batch.
            local_rank (int, optional): Local_rank for distributed training on GPUs.
                Local rank means the ranking of the current GPU device on the current
                node. Defaults to -1, which means non-distributed training.
            max_steps (int, optional): Maximum number of training steps.
                Defaults to 5e4.
            warmup_steps (int, optional): Number of steps taken to increase
                learning rate from 0 to `learning_rate`. Defaults to 2e3.
            learning_rate (float, optional):  Learning rate of the optimizer.
                Defaults to 0.002.
            weight_decay (float, optional): Weight decay to apply after each parameter
                update. Defaults to 0.01.
            adam_epsilon (float, optional): Epsilon of the AdamW optimizer.
                Defaults to 1e-8.
            max_grad_norm (float, optional): Maximum gradient norm for
                gradient clipping. Defaults to 0.
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
        checkpoint_state_dict = None
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

        self.amp = amp

        global_step = 0
        if (
            checkpoint_state_dict
            and "global_step" in checkpoint_state_dict
            and checkpoint_state_dict["global_step"]
        ):
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
            return self.processor.collate_fn(data, device, train_mode=True)

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

        get_inputs = functools.partial(
            self.processor.get_inputs, tokenizer=self.processor.tokenizer
        )
        super().fine_tune(
            train_dataloader=train_dataloader,
            get_inputs=get_inputs,
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
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            fp16=fp16,
            amp=amp,
            validation_function=validation_function,
        )

        # release GPU memories
        self.model.cpu()
        torch.cuda.empty_cache()

        self.save_model(global_step=max_steps)

    def predict(
        self,
        test_dataset,
        num_gpus=None,
        gpu_ids=None,
        local_rank=-1,
        batch_size=16,
        min_length=56,
        max_length=140,
        num_beams=4,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
        early_stopping=True,
        fp16=False,
        verbose=True,
        checkpoint=None,
        **predictor_kwargs
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
            min_length (int, optional): Minimum number of tokens in the output sequence.
                Defaults to 140.
            max_length (int, optional):  Maximum number of tokens in output
                sequence. Defaults to 150.
            num_beams (int, optional): Beam size for beam search. Defaults to 4.
            length_penalty (float, optional): Exponential penalty to the length.
                Defaults to 2.0.
            no_repeat_ngram_size (int, optional): If set to int >0, all ngrams of size
                `no_repeat_ngram_size` can only occur once in the generated summary.
                Defaults to 3.
            early_stopping (bool, optional): If set to `True` beam search is stopped
                when at least `num_beams` sentences finished per batch. Defautls to True. 
            fp16 (bool, optional): Whether to use half-precision model for prediction.
                Defaults to False.
            verbose (bool, optional): Whether to print out the training log.
                Defaults to True.
            checkpoint (str, optional):
            predictor_kwargs (dict, optional): Additional kwargs that will be forwarded
                to `Predictor`. Please consult the arguments in function 
                `PreTrainedModel::generate`.

        Returns:
            List of strings which are the summaries

        """

        device, num_gpus = get_device(
            num_gpus=num_gpus, gpu_ids=gpu_ids, local_rank=local_rank
        )
        model = move_model_to_device(self.model, device)

        checkpoint_state_dict = None
        if checkpoint:
            # checkpoint should have "model", "optimizer", "amp"
            checkpoint_state_dict = torch.load(checkpoint, map_location="cpu")
            model.load_state_dict(checkpoint_state_dict["model"])

        model.eval()

        model = parallelize_model(
            model, device, num_gpus=num_gpus, gpu_ids=gpu_ids, local_rank=local_rank,
        )

        if fp16:
            model = model.half()

        test_sampler = SequentialSampler(test_dataset)

        def collate_fn(data):
            return self.processor.collate_fn(data, device, train_mode=False)

        test_dataloader = DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )
        print("dataset length is {}".format(len(test_dataset)))

        
        predictor = Predictor(
            model,
            min_length,
            max_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            **predictor_kwargs
        )

        # move model to devices
        def this_model_move_callback(model, device):
            model = move_model_to_device(model, device)
            return parallelize_model(
                model, device, num_gpus=num_gpus, gpu_ids=gpu_ids, local_rank=local_rank
            )

        predictor = this_model_move_callback(predictor, device)

        generated_summaries = []

        for batch in tqdm(
            test_dataloader, desc="Generating summary", disable=True  # not verbose
        ):
            input_ids, masks = trim_batch(
                batch["source_ids"],
                self.tokenizer.pad_token_id,
                attention_mask=batch["source_mask"],
            )
            summaries = predictor(input_ids, masks)
            decoded_summaries = [
                self.tokenizer.decode(
                    g, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for g in summaries
            ]

            generated_summaries.extend(decoded_summaries)

        # release GPU memories
        # self.model.cpu()
        del batch
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
            full_name = os.path.join(
                output_model_dir, "abssum_{}.pt".format(self.model_name)
            )
        else:
            path, filename = os.path.split(full_name)
            print(path)
            os.makedirs(path, exist_ok=True)

        checkpoint = {
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "lr_scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "model": model_to_save.state_dict(),
            "amp": self.amp.state_dict() if self.amp else None,
            "global_step": global_step,
            "max_source_length": self.max_source_length,
            "max_target_length": self.max_target_length,
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
