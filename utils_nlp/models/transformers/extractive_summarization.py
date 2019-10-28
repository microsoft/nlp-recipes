# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from transformers.modeling_bert import (
    BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
    BertForSequenceClassification,
)
from transformers.modeling_distilbert import (
    DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
    DistilBertForSequenceClassification,
)
from transformers.modeling_roberta import (
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
    RobertaForSequenceClassification,
)
from transformers.modeling_xlnet import (
    XLNET_PRETRAINED_MODEL_ARCHIVE_MAP,
    XLNetForSequenceClassification,
)

from utils_nlp.models.transformers.common import (
    MAX_SEQ_LEN,
    TOKENIZER_CLASS,
    Transformer,
    get_device,
)

MODEL_CLASS = {}
MODEL_CLASS.update({k: BertForSequenceClassification for k in BERT_PRETRAINED_MODEL_ARCHIVE_MAP})
MODEL_CLASS.update(
    {k: RobertaForSequenceClassification for k in ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP}
)
MODEL_CLASS.update({k: XLNetForSequenceClassification for k in XLNET_PRETRAINED_MODEL_ARCHIVE_MAP})
MODEL_CLASS.update(
    {k: DistilBertForSequenceClassification for k in DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP}
)

from bertsum.prepro.data_builder import greedy_selection, combination_selection
from bertsum.prepro.data_builder import TransformerData
from utils_nlp.models.bert.extractive_text_summarization import Bunch, modified_format_to_bert, default_parameters
from bertsum.models.model_builder import Summarizer
from bertsum.models import  model_builder

class ExtSumData():
    def __init__(self, src, segs, clss, mask, mask_cls, labels=None, src_str=None, tgt_str=None):
        self.src = src
        self.segs = segs
        self.clss = clss
        self.mask = mask
        self.mask_cls = mask_cls
        self.labels = labels
        self.src_str = src_str
        self.tgt_str = tgt_str
        
class ExtSumIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, src, segs, clss, mask, mask_cls, labels=None, src_str=None, tgt_str=None):
        self.src = src
        self.segs = segs
        self.clss = clss
        self.mask = mask
        self.mask_cls = mask_cls
        self.labels = labels
        self.src_str = src_str
        self.tgt_str = tgt_str
    
    def __iter__(self):
        if self.labels is not None:
            return iter(zip(self.src, self.segs, self.clss, \
                self.mask, self.mask_cls,  self.src_str, self.labels, self.tgt_str))
        else:
            return  iter(zip(self.src, self.segs, self.clss, \
                self.mask, self.mask_cls, self.src_str))
        

    def __getitem__(self, index):
        if self.labels is not None:
            return ExtSumData(self.src[index], self.segs[index], self.clss[index], \
                self.mask[index], self.mask_cls[index],   self.labels[index], self.src_str[index], self.tgt_str[index])
        else:
            return ExtSumData(self.src[index], self.segs[index], self.clss[index], \
                self.mask[index], self.mask_cls[index], None, self.src_str[index], None)

    def __len__(self):
        return len(self.src)

    

class ExtSumProcessor:    
    def __init__(self, model_name="bert-base-cased", to_lower=False, cache_dir="."):
        self.tokenizer = TOKENIZER_CLASS[model_name].from_pretrained(
            model_name, do_lower_case=to_lower, cache_dir=cache_dir
        )


        default_preprocessing_parameters = {
            "max_nsents": 200,
            "max_src_ntokens": 2000,
            "min_nsents": 3,
            "min_src_ntokens": 5,
            "use_interval": True,
        }
        args = Bunch(default_preprocessing_parameters)
        self.preprossor = TransformerData(args, self.tokenizer)
        
        
    @staticmethod
    def get_inputs(batch, model_name, train_mode=True):
        if model_name.split("-")[0] in ["bert", "xlnet", "roberta", "distilbert"]:
            if train_mode:
               # return {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
            #src, segs, clss, mask, mask_cls, src_str
            # labels must be the last
                return {"x": batch.src, "segs": batch.segs, "clss": batch.clss,
                        "mask": batch.mask, "mask_cls": batch.mask_cls, "labels": batch.labels}
            else:
                return {"x": batch.src, "segs": batch.segs, "clss": batch.clss,
                        "mask": batch.mask, "mask_cls": batch.mask_cls}
        else:
            raise ValueError("Model not supported: {}".format(model_name))

    @staticmethod
    def get_inputs_2(batch, model_name, train_mode=True):
        if model_name.split("-")[0] in ["bert", "xlnet", "roberta", "distilbert"]:
            if train_mode:
               # return {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
            #src, segs, clss, mask, mask_cls, src_str
            # labels must be the last
                return {"x": batch[0], "segs": batch[1], "clss": batch[2],
                        "mask": batch[3], "mask_cls": batch[4], "labels": batch[5]}
            else:
                return {"x": batch[0], "segs": batch[1], "clss": batch[2],
                       "mask": batch[3], "mask_cls": batch[4]}
        else:
            raise ValueError("Model not supported: {}".format(model_name))
            
    def preprocess(self, sources, targets=None, oracle_mode="greedy", selections=3):
        """preprocess multiple data points"""
        
        is_labeled = False
        if targets is None:
            for source in sources:
                yield list(self._preprocess_single(source, None, oracle_mode, selections))
        else:
            for (source, target) in zip(sources, targets):
                yield list(self._preprocess_single(source, target, oracle_mode, selections))
            is_labeled = True
            
        
    def preprocess_3(self, sources, targets=None, oracle_mode="greedy", selections=3):
        """preprocess multiple data points"""
        
        is_labeled = False
        if targets is None:
            data = [self._preprocess_single(source, None, oracle_mode, selections) for source in sources]     
        else:
            data = [self._preprocess_single(source, target, oracle_mode, selections) for (source, target) in zip(sources, targets)]
            is_labeled = True
            
        def _pad(data, pad_id, width=-1):
            if (width == -1):
                width = max(len(d) for d in data)
            rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
            return rtn_data

    
        if data is not None:
            pre_src = [x[0] for x in data]
            pre_segs = [x[2] for x in data]
            pre_clss = [x[3] for x in data]

            src = torch.tensor(_pad(pre_src, 0))
        
            pre_labels = None
            labels = None
            if is_labeled:
                pre_labels = [x[1] for x in data]
                labels = torch.tensor(_pad(pre_labels, 0))
            segs = torch.tensor(_pad(pre_segs, 0))
            #mask = 1 - (src == 0)
            mask = ~(src == 0)

            clss = torch.tensor(_pad(pre_clss, -1))
            #mask_cls = 1 - (clss == -1)
            mask_cls = ~(clss == -1)
            clss[clss == -1] = 0

            #setattr(self, 'clss', clss.to(device))
            #setattr(self, 'mask_cls', mask_cls.to(device))
            #setattr(self, 'src', src.to(device))
            #setattr(self, 'segs', segs.to(device))
            #setattr(self, 'mask', mask.to(device))
            src_str = [x[-2] for x in data]
            #setattr(self, 'src_str', src_str)
            #x, segs, clss, mask, mask_cls, 
            #td = TensorDataset(src, segs, clss, mask, mask_cls)
            #td = src, segs, clss, mask, mask_cls, None, src_str, None
            td = ExtSumIterableDataset(src, segs, clss, mask, mask_cls, None, src_str, None)
            if is_labeled:
                #setattr(self, 'labels', labels.to(device))
                tgt_str = [x[-1] for x in data]
                #setattr(self, 'tgt_str', tgt_str)
                #td = TensorDataset(src, segs, clss, mask, mask_cls, labels)
                td = ExtSumIterableDataset(src, segs, clss, mask, mask_cls, labels, src_str, tgt_str)
            return td
        

    def preprocess_2(self, sources, targets=None, oracle_mode="greedy", selections=3):
        """preprocess multiple data points"""
        
        is_labeled = False
        if targets is None:
            data = [self._preprocess_single(source, None, oracle_mode, selections) for source in sources]     
        else:
            data = [self._preprocess_single(source, target, oracle_mode, selections) for (source, target) in zip(sources, targets)]
            is_labeled = True
            
        def _pad(data, pad_id, width=-1):
            if (width == -1):
                width = max(len(d) for d in data)
            rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
            return rtn_data

    
        if data is not None:
            pre_src = [x[0] for x in data]
            pre_segs = [x[2] for x in data]
            pre_clss = [x[3] for x in data]

            src = torch.tensor(_pad(pre_src, 0))
        
            pre_labels = None
            labels = None
            if is_labeled:
                pre_labels = [x[1] for x in data]
                labels = torch.tensor(_pad(pre_labels, 0))
            segs = torch.tensor(_pad(pre_segs, 0))
            #mask = 1 - (src == 0)
            mask = ~(src == 0)

            clss = torch.tensor(_pad(pre_clss, -1))
            #mask_cls = 1 - (clss == -1)
            mask_cls = ~(clss == -1)
            clss[clss == -1] = 0

            #setattr(self, 'clss', clss.to(device))
            #setattr(self, 'mask_cls', mask_cls.to(device))
            #setattr(self, 'src', src.to(device))
            #setattr(self, 'segs', segs.to(device))
            #setattr(self, 'mask', mask.to(device))
            src_str = [x[-2] for x in data]
            #setattr(self, 'src_str', src_str)
            #x, segs, clss, mask, mask_cls, 
            td = TensorDataset(src, segs, clss, mask, mask_cls)
            if (is_labeled):
                #setattr(self, 'labels', labels.to(device))
                tgt_str = [x[-1] for x in data]
                #setattr(self, 'tgt_str', tgt_str)
                td = TensorDataset(src, segs, clss, mask, mask_cls, labels)
            return td
                
        
    
    def _preprocess_single(self, source, target=None, oracle_mode="greedy", selections=3):
        """preprocess single data point"""
        oracle_ids = None
        if target is not None:
            if (oracle_mode == 'greedy'):
                oracle_ids = greedy_selection(source, target, selections)
            elif (oracle_mode == 'combination'):
                oracle_ids = combination_selection(source, target, selections)
                print(oracle_ids)
            
         
        b_data  = self.preprossor.preprocess(source, target, oracle_ids)
        
        if (b_data is None):
            return None
        indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        return (indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt)
        #return {"src": indexed_tokens, "labels": labels, "segs": segments_ids, 'clss': cls_ids,
        #           'src_txt': src_txt, "tgt_txt": tgt_txt}

from transformers import PreTrainedModel,  PretrainedConfig, DistilBertModel, BertModel, DistilBertConfig
import bertsum.distributed as distributed

class ExtractiveSummarizer(Transformer):
    def __init__(self, model_name="distilbert-base-uncased", model_class=DistilBertModel, cache_dir="."):
        super().__init__(
            model_class=MODEL_CLASS,
            model_name=model_name,
            num_labels=0,
            cache_dir=cache_dir,
        )
        args = Bunch(default_parameters)
        self.model = Summarizer("transformer", args, model_class, model_name, None, cache_dir)
        
        
    @staticmethod
    def list_supported_models():
        return list(MODEL_CLASS)

    def fit(
        self,
        train_data_iterator,
        device="cuda",
        num_epochs=1,
        batch_size=32,
        num_gpus=None,
        local_rank=-1,
        weight_decay=0.0,
        learning_rate=1e-4,
        adam_epsilon=1e-8,
        warmup_steps=10000,
        verbose=True,
        seed=None,
        decay_method='noam', 
        lr=0.002,
        accum_count=2,
        **kwargs
    ):
        #device, num_gpus = get_device(device=device, num_gpus=num_gpus, local_rank=local_rank)
        device = torch.device("cuda:{}".format(0))
        gpu_rank = distributed.multi_init(0, 1, "0")
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        
        self.model.to(device)
        
        args = Bunch(default_parameters)
        optim = model_builder.build_optim(args, self.model, None)
        
        super().fine_tune(
            train_data_iterator,
            get_inputs=ExtSumProcessor.get_inputs,
            device=device,
            optimizer=optim,
            per_gpu_train_batch_size=batch_size,
            n_gpu=num_gpus,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            adam_epsilon=adam_epsilon,
            warmup_steps=warmup_steps,
            verbose=verbose,
            seed=seed,
            **kwargs,
        )

    def predict(self, eval_data_iterator, device, batch_size=16, sentence_seperator="<q>", top_n=3, block_trigram=True, num_gpus=1, verbose=True, cal_lead=False):
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s))>0:
                    return True
            return False
        def _get_pred(batch, sent_scores):
            #return sent_scores
            if cal_lead:
                selected_ids = list(range(batch.clss.size(1)))*len(batch.clss)
            else:
                #negative_sent_score = [-i for i in sent_scores[0]]
                selected_ids = np.argsort(-sent_scores, 1)   
                # selected_ids = np.sort(selected_ids,1)
            pred = []
            for i, idx in enumerate(selected_ids):
                _pred = []
                if(len(batch.src_str[i])==0):
                    pred.append('')
                    continue
                for j in selected_ids[i][:len(batch.src_str[i])]:
                    if(j>=len( batch.src_str[i])):
                        continue
                    candidate = batch.src_str[i][j].strip()
                    if(block_trigram):
                        if(not _block_tri(candidate,_pred)):
                            _pred.append(candidate)
                    else:
                        _pred.append(candidate)

                    # only select the top 3
                    if len(_pred) == top_n:
                        break

                #_pred = '<q>'.join(_pred)
                _pred = sentence_seperator.join(_pred)  
                pred.append(_pred.strip())
            return pred
        
         #for batch in tqdm(eval_data_iterator, desc="Evaluating", disable=not verbose):
        self.model.eval()
        pred = []
        for batch in eval_data_iterator:
            batch = batch.to(device)
            #batch = tuple(t.to(device) for t in batch)
            #batch = tuple(t.to(device) for t in batch if type(t)==torch.Tensor)
            with torch.no_grad():
                inputs = ExtSumProcessor.get_inputs(batch, self.model_name, train_mode=False)
                outputs = self.model(**inputs)
                sent_scores = outputs[0]
                sent_scores = sent_scores.detach().cpu().numpy()
                #return sent_scores
                pred.extend(_get_pred(batch, sent_scores))
            #yield logits.detach().cpu().numpy()
        return pred


                    
        """preds = list(
            super().predict(
                eval_dataset=eval_dataset,
                get_inputs=ExtSumProcessor.get_inputs,
                device=device,
                per_gpu_eval_batch_size=batch_size,
                n_gpu=num_gpus,
                verbose=True,
            )
        )
        preds = np.concatenate(preds)
        # todo generator & probs
        return np.argmax(preds, axis=1)
        """