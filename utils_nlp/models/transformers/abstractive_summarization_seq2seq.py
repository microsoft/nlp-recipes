import os
from datetime import datetime
import jsonlines
import logging
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler

from transformers import BertTokenizer, RobertaTokenizer
from transformers.modeling_roberta import ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_xlm_roberta import XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers import RobertaConfig, BertConfig, DistilBertConfig, XLMRobertaConfig
from transformers import (
    BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
    DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
)


from utils_nlp.models.transformers.common import TOKENIZER_CLASS, Transformer
from utils_nlp.common.pytorch_utils import get_device, move_model_to_device, parallelize_model
from s2s_ft.utils import load_and_cache_examples, Seq2seqDatasetForBert, batch_list_to_batch_tensors
from s2s_ft.modeling import BertForSequenceToSequence
from s2s_ft.modeling import UNILM_PRETRAINED_MODEL_ARCHIVE_MAP
from s2s_ft.tokenization_unilm import UnilmTokenizer
from s2s_ft.configuration_unilm import UnilmConfig, UNILM_PRETRAINED_CONFIG_ARCHIVE_MAP
from s2s_ft.config import BertForSeq2SeqConfig
import s2s_ft.s2s_loader as seq2seq_loader
from s2s_ft.modeling_decoding import BertForSeq2SeqDecoder

MODEL_CLASS = {}
MODEL_CLASS.update({k: BertForSequenceToSequence for k in BERT_PRETRAINED_MODEL_ARCHIVE_MAP})
MODEL_CLASS.update({k: BertForSequenceToSequence for k in ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP})
MODEL_CLASS.update({k: BertForSequenceToSequence for k in XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP})
MODEL_CLASS.update({k: BertForSequenceToSequence for k in UNILM_PRETRAINED_MODEL_ARCHIVE_MAP})

TOKENIZER_CLASS.update({k: UnilmTokenizer for k in UNILM_PRETRAINED_CONFIG_ARCHIVE_MAP})

CONFIG_CLASS = {}
CONFIG_CLASS.update({k: BertConfig for k in BERT_PRETRAINED_CONFIG_ARCHIVE_MAP})
CONFIG_CLASS.update({k: RobertaConfig for k in ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP})
CONFIG_CLASS.update({k: DistilBertConfig for k in DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP})
CONFIG_CLASS.update({k: XLMRobertaConfig for k in XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP})
CONFIG_CLASS.update({k: UnilmConfig for k in UNILM_PRETRAINED_CONFIG_ARCHIVE_MAP})

logger = logging.getLogger(__name__)


def _get_model_type(model_name):
    if "-".join(model_name.split("-")[:2]) == "xlm-roberta":
        return "xlm-roberta"
    else:
        return model_name.split("-")[0]


def _get_decode_tokenizer(model_type, bert_model_name, to_lower, max_seq_len):
    if model_type == "roberta":
        decode_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    else:
        decode_tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=to_lower)
    decode_tokenizer.max_len = max_seq_len

    return decode_tokenizer


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith("##") and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


class S2SAbsSumDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, idx):
        return self.features[idx]

    def __len__(self):
        return len(self.features)


class S2SAbsSumProcessor:
    def __init__(
        self,
        model_name="unilm-base-cased",
        to_lower=False,
        max_seq_len=512,
        cache_dir=".",
        cached_features_file_name="train_features",
    ):

        self.tokenizer = TOKENIZER_CLASS[model_name].from_pretrained(
            model_name, do_lower_case=to_lower, cache_dir=cache_dir
        )

        self.cached_features_file = os.path.join(cache_dir, cached_features_file_name)

        self._model_name = model_name
        self._bert_model_name = self._model_name.replace("unilm", "bert")
        self._model_type = _get_model_type(self._model_name)

        self.decode_tokenizer = _get_decode_tokenizer(
            model_type=self._model_type,
            bert_model_name=self._bert_model_name,
            to_lower=to_lower,
            max_seq_len=max_seq_len,
        )

    @classmethod
    def get_inputs(cls, batch, device, model_name):
        batch = tuple(t.to(device) for t in batch)
        inputs = {
            "source_ids": batch[0],
            "target_ids": batch[1],
            "pseudo_ids": batch[2],
            "num_source_tokens": batch[3],
            "num_target_tokens": batch[4],
        }
        return inputs

    def train_dataset_from_iterable_sum_ds(
        self, sum_ds, load_cached_features=True, local_rank=-1, keep_train_file=False
    ):
        # If in distributed mode, block all processes except GPU 0
        if local_rank not in [-1, 0]:
            torch.distributed.barrier()

        temp_dir = "./"
        temp_train_file = os.path.join(
            temp_dir, "train_file_" + datetime.now().strftime("%m%d%Y%H%M%S") + ".json"
        )
        try:
            with jsonlines.open(temp_train_file, mode="w") as writer:
                for source, target in zip(sum_ds, sum_ds.get_target()):
                    writer.write({"src": source, "tgt": target})

            train_dataset = self.train_dataset_from_file(
                train_file=temp_train_file,
                load_cached_features=load_cached_features,
                local_rank=local_rank,
            )

        finally:
            if not keep_train_file and os.path.exists(temp_train_file):
                os.remove(temp_train_file)
        # All processes entered torch.distributed.barrier() after GPU 0 joined,
        # so all processes are unblocked
        if local_rank == 0:
            torch.distributed.barrier()
        return train_dataset

    def train_dataset_from_sum_ds(
        self, sum_ds, load_cached_features=True, local_rank=-1, keep_train_file=False
    ):
        # If in distributed mode, block all processes except GPU 0
        if local_rank not in [-1, 0]:
            torch.distributed.barrier()

        temp_dir = "./"
        temp_train_file = os.path.join(
            temp_dir, "train_file_" + datetime.now().strftime("%m%d%Y%H%M%S") + ".json"
        )
        try:
            with jsonlines.open(temp_train_file, mode="w") as writer:
                for item in sum_ds:
                    writer.write(item)

            train_dataset = self.train_dataset_from_file(
                train_file=temp_train_file,
                load_cached_features=load_cached_features,
                local_rank=local_rank,
            )

        finally:
            if not keep_train_file and os.path.exists(temp_train_file):
                os.remove(temp_train_file)
        # All processes entered torch.distributed.barrier() after GPU 0 joined,
        # so all processes are unblocked
        if local_rank == 0:
            torch.distributed.barrier()
        return train_dataset

    def train_dataset_from_file(self, train_file, load_cached_features=False, local_rank=-1):
        if not load_cached_features and os.path.exists(self.cached_features_file):
            logger.info("Deleting cached feature file {}".format(self.cached_features_file))
            os.remove(self.cached_features_file)

        train_features = load_and_cache_examples(
            example_file=train_file,
            tokenizer=self.tokenizer,
            local_rank=local_rank,
            cached_features_file=self.cached_features_file,
            shuffle=True,
        )

        return S2SAbsSumDataset(train_features)

    def test_dataset_from_iterable_sum_ds():
        pass

    def test_dataset_from_sum_ds(self, sum_ds):
        input_lines = []
        for example in sum_ds:
            input_lines.append(self._preprocess_test_src(example))

        input_lines = sorted(list(enumerate(input_lines)), key=lambda x: -len(x[1]))
        return S2SAbsSumDataset(input_lines)

    def test_dataset_from_file(self, test_file):
        with open(test_file, encoding="utf-8", mode="r") as fin:
            input_lines = []
            for line in fin:
                example = json.loads(line)
                input_lines.append(self._preprocess_test_src(example))

        input_lines = sorted(list(enumerate(input_lines)), key=lambda x: -len(x[1]))
        return S2SAbsSumDataset(input_lines)

    def _preprocess_test_src(self, example):
        if isinstance(example["src"], list):
            source_tokens = example["src"]
        else:
            source_tokens = self.decode_tokenizer.tokenize(example["src"])

        if self._model_type != "roberta":
            enter_token = self.decode_tokenizer.tokenize("Enter\nToken")[1]
            source_tokens = [enter_token if x == "[X_SEP]" else x for x in source_tokens]

        return source_tokens


class S2SConfig:
    def __init__(
        self,
        new_segment_ids=False,
        new_pos_ids=False,
        forbid_duplicate_ngrams=False,
        forbid_ignore_word=None,
        min_len=1,
        ngram_size=3,
        mode="s2s",
        s2s_special_token=False,
        s2s_add_segment=False,
        s2s_share_segment=False,
        pos_shift=False,
        ffn_type=0,
        num_qkv=0,
        seg_emb=False,
    ):

        self.new_segment_ids = new_segment_ids
        self.new_segment_ids = new_pos_ids
        self.forbid_duplicate_ngrams = forbid_duplicate_ngrams
        self.forbid_ignore_word = forbid_ignore_word
        self.min_len = min_len
        self.ngram_size = ngram_size
        self.mode = mode
        self.s2s_special_token = s2s_special_token
        self.s2s_add_segment = s2s_add_segment
        self.s2s_share_segment = s2s_share_segment
        self.pos_shift = pos_shift
        self.ffn_type = ffn_type
        self.num_qkv = num_qkv
        self.seg_emb = seg_emb


class S2SAbstractiveSummarizer(Transformer):
    def __init__(
        self,
        model_name="unilm-base-cased",
        to_lower=False,
        cache_dir=".",
        load_model_from_dir=None,
        model_file_name=None,
        label_smoothing=0.1,
        max_seq_len=512,
        *model_args,
        **kwargs
    ):

        if model_name not in self.list_supported_models():
            raise ValueError(
                "Model name {0} is not supported by {1}. "
                "Call '{1}.list_supported_models()' to get all supported model "
                "names.".format(value, self.__class__.__name__)
            )
        model_class = MODEL_CLASS[model_name]
        config_class = CONFIG_CLASS[model_name]

        self._model_name = model_name
        ## Check with MSRA team how to determine tokenizer and config from model names
        self._bert_model_name = self._model_name.replace("unilm", "bert")
        self.cache_dir = cache_dir
        self.load_model_from_dir = load_model_from_dir
        self.do_lower_case = to_lower
        self.max_seq_length = max_seq_len

        self._model_type = _get_model_type(model_name)

        if load_model_from_dir is None:
            model_to_load = self._model_name
        elif model_file_name is None:
            logger.info("Loading cached model from {}".format(load_model_from_dir))
            model_to_load = load_model_from_dir
        else:
            model_to_load = os.path.join(load_model_from_dir, model_file_name)
            logger.info("Loading cached model from {}".format(model_to_load))

        # TODO: double check
        model_config = config_class.from_pretrained(self._model_name, cache_dir=cache_dir)
        config = BertForSeq2SeqConfig.from_exist_config(
            config=model_config, label_smoothing=label_smoothing
        )
        logger.info("Model config for seq2seq: %s", str(config))

        self.model = model_class.from_pretrained(
            model_to_load, config=config, model_type=self._model_type, cache_dir=cache_dir
        )

        self.tokenizer = TOKENIZER_CLASS[model_name].from_pretrained(
            self._model_name, do_lower_case=to_lower, cache_dir=cache_dir, output_loading_info=False
        )

        self.decode_tokenizer = _get_decode_tokenizer(
            model_type=self._model_type,
            bert_model_name=self._bert_model_name,
            to_lower=to_lower,
            max_seq_len=max_seq_len,
        )

    @staticmethod
    def list_supported_models():
        return list(MODEL_CLASS)

    def fit(
        self,
        train_dataset,
        max_source_seq_length=464,
        max_target_seq_length=48,
        learning_rate=5e-5,
        per_gpu_batch_size=8,
        num_epochs=1,
        recover_step=-1,
        max_steps=-1,
        local_rank=-1,
        num_gpus=None,
        gpu_ids=None,
        gradient_accumulation_steps=1,
        weight_decay=0.01,
        adam_epsilon=1e-8,
        warmup_steps=0,
        fp16=False,
        fp16_opt_level="O1",
        max_grad_norm=1.0,
        save_model=True,
        verbose=True,
        seed=None,
        random_prob=0.1,
        keep_prob=0.1,
    ):
        # Before we do anything with models, we want to ensure that we get fp16 execution
        # of torch.einsum if args.fp16 is set. Otherwise it'll default to "promote" mode,
        # and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
        # remove the need for this code, but it is still valid.
        if fp16:
            try:
                from apex import amp

                amp.register_half_function(torch, "einsum")
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex")
        else:
            amp = None

        # get device
        device, num_gpus = get_device(num_gpus=num_gpus, gpu_ids=gpu_ids, local_rank=local_rank)

        # move model
        self.model = move_model_to_device(model=self.model, device=device)

        # if gpu_ids is not None:
        #     per_node_train_batch_size = per_gpu_batch_size * len(gpu_ids)
        # elif num_gpus is not None and torch.cuda.is_available():
        #     per_node_train_batch_size = per_gpu_batch_size * min(
        #         num_gpus, torch.cuda.device_count())
        # else:
        #     per_node_train_batch_size = per_gpu_batch_size * max(1, torch.cuda.device_count())
        # per_node_train_batch_size = per_node_train_batch_size * gradient_accumulation_steps

        per_node_train_batch_size = (
            per_gpu_batch_size * max(1, num_gpus) * gradient_accumulation_steps
        )

        # actual batch size, i.e. number of samples between each parameter update
        batch_size = per_node_train_batch_size * (
            torch.distributed.get_world_size() if local_rank != -1 else 1
        )

        # max_steps is mainly used by the scheduler to determine the learning rate,
        # together with global_step
        if max_steps == -1:
            max_steps = max(num_epochs * len(train_dataset) // batch_size, 1)

        # init optimizer and scheduler
        self.optimizer = Transformer.get_default_optimizer(
            self.model, weight_decay, learning_rate, adam_epsilon
        )

        if fp16:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=fp16_opt_level
            )

        global_step = 0
        if recover_step > 0:
            model_recover_checkpoint = os.path.join(
                self.load_model_from_dir, "model.{}.bin".format(recover_step)
            )
            logger.info(" ** Recover model checkpoint in %s ** ", model_recover_checkpoint)
            model_state_dict = torch.load(model_recover_checkpoint, map_location="cpu")
            optimizer_recover_checkpoint = os.path.join(
                args.output_dir, "optim.{}.bin".format(recover_step)
            )
            checkpoint_state_dict = torch.load(optimizer_recover_checkpoint, map_location="cpu")

            self.optimizer.load_state_dict(checkpoint_state_dict["optimizer"])
            self.model.load_state_dict(model_state_dict)

            if fp16:
                amp.load_state_dict(checkpoint_state_dict["amp"])

            global_step = recover_step

        if max_steps <= global_step:
            logger.info("Training is done. Please use a new dir or clean this dir!")

        self.scheduler = Transformer.get_default_scheduler(
            optimizer=self.optimizer, warmup_steps=warmup_steps, num_training_steps=max_steps
        )

        if recover_step > 0:
            self.scheduler.load_state_dict(checkpoint_state_dict["lr_scheduler"])

        self.model = parallelize_model(
            model=self.model,
            device=device,
            num_gpus=num_gpus,
            gpu_ids=gpu_ids,
            local_rank=local_rank,
        )

        train_dataset = Seq2seqDatasetForBert(
            features=train_dataset,
            max_source_len=max_source_seq_length,
            max_target_len=max_target_seq_length,
            vocab_size=self.tokenizer.vocab_size,
            cls_id=self.tokenizer.cls_token_id,
            sep_id=self.tokenizer.sep_token_id,
            pad_id=self.tokenizer.pad_token_id,
            mask_id=self.tokenizer.mask_token_id,
            random_prob=random_prob,
            keep_prob=keep_prob,
            num_training_instances=batch_size * max_steps,
            offset=batch_size * global_step,
        )

        # The training features are shuffled
        train_sampler = (
            SequentialSampler(train_dataset)
            if local_rank == -1
            else DistributedSampler(train_dataset, shuffle=False)
        )
        # batch_size of the dataloader is the number of samples to load each iteration on each node
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=per_node_train_batch_size // gradient_accumulation_steps,
            collate_fn=batch_list_to_batch_tensors,
        )

        global_step, _ = super().fine_tune(
            train_dataloader=train_dataloader,
            device=device,
            num_gpus=num_gpus,
            get_inputs=S2SAbsSumProcessor.get_inputs,
            max_steps=max_steps,
            global_step=global_step,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            local_rank=local_rank,
            fp16=fp16,
            fp16_opt_level=fp16_opt_level,
            amp=amp,
            max_grad_norm=max_grad_norm,
            verbose=verbose,
            seed=seed,
        )

        if save_model:
            self.save_model(global_step, fp16)

    def predict(
        self,
        test_dataset,
        batch_size=4,
        max_tgt_length=128,
        beam_size=1,
        need_score_traces=False,
        length_penalty=0,
        s2s_config=S2SConfig(),
        num_gpus=None,
        gpu_ids=None,
        local_rank=-1,
        fp16=False,
        amp=False,
        seed=None,
        verbose=True,
    ):
        if need_score_traces and beam_size <= 1:
            raise ValueError("Score trace is only available for beam search with beam size > 1.")
        if max_tgt_length >= self.max_seq_length - 2:
            raise ValueError("Maximum tgt length exceeds max seq length - 2.")

        if self._model_type == "roberta":
            is_roberta = True
            no_segment_embedding = True
            vocab = self.decode_tokenizer.encoder
        else:
            is_roberta = False
            no_segment_embedding = False
            vocab = self.decode_tokenizer.vocab

        cls_token = "<s>" if is_roberta else "[CLS]"
        sep_token = "</s>" if is_roberta else "[SEP]"
        pad_token = "<pad>" if is_roberta else "[PAD]"
        mask_token = "<mask>" if is_roberta else "[MASK]"

        max_src_length = self.max_seq_length - 2 - max_tgt_length
        bi_uni_pipeline = []
        bi_uni_pipeline.append(
            seq2seq_loader.Preprocess4Seq2seqDecoder(
                list(vocab.keys()),
                self.decode_tokenizer.convert_tokens_to_ids,
                self.max_seq_length,
                max_tgt_length=max_tgt_length,
                new_segment_ids=s2s_config.new_segment_ids,
                mode=s2s_config.mode,
                num_qkv=s2s_config.num_qkv,
                s2s_special_token=s2s_config.s2s_special_token,
                s2s_add_segment=s2s_config.s2s_add_segment,
                s2s_share_segment=s2s_config.s2s_share_segment,
                pos_shift=s2s_config.pos_shift,
                cls_token=cls_token,
                sep_token=sep_token,
                pad_token=pad_token,
            )
        )

        def collate_fn(input_batch):
            buf_id = [x[0] for x in input_batch]
            buf = [x[1][:max_src_length] for x in input_batch]
            max_a_len = max([len(x) for x in buf])
            instances = []
            for instance in [(x, max_a_len) for x in buf]:
                for proc in bi_uni_pipeline:
                    instance = proc(instance)
                instances.append(instance)
            batch = seq2seq_loader.batch_list_to_batch_tensors(instances)

            return (batch, buf_id)

        pair_num_relation = 0
        cls_num_labels = 2
        type_vocab_size = (
            6 + (1 if s2s_config.s2s_add_segment else 0) if s2s_config.new_segment_ids else 2
        )
        mask_word_id, eos_word_ids, sos_word_id = self.decode_tokenizer.convert_tokens_to_ids(
            [mask_token, sep_token, sep_token]
        )
        forbid_ignore_set = None
        if s2s_config.forbid_ignore_word:
            w_list = []
            for w in args.forbid_ignore_word.split("|"):
                if w.startswith("[") and w.endswith("]"):
                    w_list.append(w.upper())
                else:
                    w_list.append(w)
            forbid_ignore_set = set(self.decode_tokenizer.convert_tokens_to_ids(w_list))

        if hasattr(self.model, "module"):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        model = BertForSeq2SeqDecoder.from_pretrained(
            self._bert_model_name,
            state_dict=state_dict,
            num_labels=cls_num_labels,
            num_rel=pair_num_relation,
            type_vocab_size=type_vocab_size,
            task_idx=3,
            mask_word_id=mask_word_id,
            search_beam_size=beam_size,
            length_penalty=length_penalty,
            eos_id=eos_word_ids,
            sos_id=sos_word_id,
            forbid_duplicate_ngrams=s2s_config.forbid_duplicate_ngrams,
            forbid_ignore_set=forbid_ignore_set,
            ngram_size=s2s_config.ngram_size,
            min_len=s2s_config.min_len,
            mode=s2s_config.mode,
            max_position_embeddings=self.max_seq_length,
            ffn_type=s2s_config.ffn_type,
            num_qkv=s2s_config.num_qkv,
            seg_emb=s2s_config.seg_emb,
            pos_shift=s2s_config.pos_shift,
            is_roberta=is_roberta,
            no_segment_embedding=no_segment_embedding,
        )

        del state_dict

        if fp16:
            model.half()
        # get device
        device, num_gpus = get_device(num_gpus=num_gpus, gpu_ids=gpu_ids, local_rank=local_rank)
        # move model
        model = move_model_to_device(model=model, device=device)

        batch_size = batch_size * max(1, num_gpus)

        model = parallelize_model(
            model=model, device=device, num_gpus=num_gpus, gpu_ids=gpu_ids, local_rank=local_rank
        )

        torch.cuda.empty_cache()
        model.eval()
        first_batch = True
        batch_count = 0

        output_lines = [""] * len(test_dataset)
        score_trace_list = [None] * len(test_dataset)

        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(
            test_dataset, sampler=test_sampler, batch_size=batch_size, collate_fn=collate_fn
        )
        for batch, buf_id in tqdm(test_dataloader, desc="Evaluating", disable=not verbose):
            batch_count += 1
            with torch.no_grad():
                batch = [t.to(device) if t is not None else None for t in batch]
                input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = batch
                traces = model(
                    input_ids,
                    token_type_ids,
                    position_ids,
                    input_mask,
                    task_idx=task_idx,
                    mask_qkv=mask_qkv,
                )
                if beam_size > 1:
                    traces = {k: v.tolist() for k, v in traces.items()}
                    output_ids = traces["pred_seq"]
                else:
                    output_ids = traces.tolist()
                print(len(output_ids))
                print(len(batch[0]))
                for i in range(len(batch[0])):
                    w_ids = output_ids[i]
                    output_buf = self.decode_tokenizer.convert_ids_to_tokens(w_ids)
                    output_tokens = []
                    for t in output_buf:
                        if t in (sep_token, pad_token):
                            break
                        output_tokens.append(t)
                    if is_roberta:
                        output_sequence = self.decode_tokenizer.convert_tokens_to_string(
                            output_tokens
                        )
                    else:
                        output_sequence = " ".join(detokenize(output_tokens))
                    if "\n" in output_sequence:
                        output_sequence = " [X_SEP] ".join(output_sequence.split("\n"))
                    output_lines[buf_id[i]] = output_sequence
                    if first_batch or batch_count % 50 == 0:
                        logger.info("{} = {}".format(buf_id[i], output_sequence))
                    if need_score_traces:
                        score_trace_list[buf_id[i]] = {
                            "scores": traces["scores"][i],
                            "wids": traces["wids"][i],
                            "ptrs": traces["ptrs"][i],
                        }

            first_batch = False
        if need_score_traces:
            return output_lines, score_trace_list
        else:
            return output_lines

    def save_model(self, global_step, fp16):
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(
            model_to_save.state_dict(),
            os.path.join(self.cache_dir, "model.{}.bin".format(global_step)),
        )
        optim_to_save = {
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.scheduler.state_dict(),
        }
        if fp16:
            optim_to_save["amp"] = self.amp_state_dict
        torch.save(optim_to_save, os.path.join(self.cache_dir, "optim.{}.bin".format(global_step)))
