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
from utils_nlp.models.bert.extractive_text_summarization import Bunch, default_parameters
from bertsum.models.model_builder import Summarizer
from bertsum.models import  model_builder
from transformers import PreTrainedModel,  PretrainedConfig, DistilBertModel, BertModel, DistilBertConfig
import bertsum.distributed as distributed
import time

import itertools

from bertsum.models.data_loader  import DataIterator
from bertsum.models import  model_builder, data_loader
class Bunch(object):
    """ Class which convert a dictionary to an object """

    def __init__(self, adict):
        self.__dict__.update(adict)

def get_dataset(file_list, is_train=False):
        if is_train:
            random.shuffle(file_list)
        for file in file_list:
            yield torch.load(file)
            
def get_data_loader(file_list, device, is_labeled=False, batch_size=3000):
    """
    Function to get data iterator over a list of data objects.

    Args:
        dataset (list of objects): a list of data objects.
        is_test (bool): it specifies whether the data objects are labeled data.
        batch_size (int): number of tokens per batch.
        
    Returns:
        DataIterator

    """
    args = Bunch({})
    args.use_interval = True
    args.batch_size = batch_size
    data_iter = None
    data_iter  = data_loader.Dataloader(args, get_dataset(file_list), args.batch_size, device, shuffle=False, is_test=is_labeled)
    return data_iter
            

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
        if model_name.split("-")[0] in ["bert", "distilbert"]:
            if train_mode:
            # labels must be the last
                return {"x": batch.src, "segs": batch.segs, "clss": batch.clss,
                        "mask": batch.mask, "mask_cls": batch.mask_cls, "labels": batch.labels}
            else:
                return {"x": batch.src, "segs": batch.segs, "clss": batch.clss,
                        "mask": batch.mask, "mask_cls": batch.mask_cls}
        else:
            raise ValueError("Model not supported: {}".format(model_name))

            
    def preprocess(self, sources, targets=None, oracle_mode="greedy", selections=3):
        """preprocess multiple data points"""
        
        is_labeled = False
        if targets is None:
            for source in sources:
                yield self._preprocess_single(source, None, oracle_mode, selections)
        else:
            for (source, target) in zip(sources, targets):
                yield self._preprocess_single(source, target, oracle_mode, selections)
            is_labeled = True
            
    def _preprocess_single(self, source, target=None, oracle_mode="greedy", selections=3):
        """preprocess single data point"""
        oracle_ids = None
        if target is not None:
            if (oracle_mode == 'greedy'):
                oracle_ids = greedy_selection(source, target, selections)
            elif (oracle_mode == 'combination'):
                oracle_ids = combination_selection(source, target, selections)
                
        b_data  = self.preprossor.preprocess(source, target, oracle_ids)
        if (b_data is None):
            return None
        indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        return {"src": indexed_tokens, "labels": labels, "segs": segments_ids, 'clss': cls_ids,
                   'src_txt': src_txt, "tgt_txt": tgt_txt}

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
        train_data_iterator_function,
        device="cuda",
        num_gpus=None,
        local_rank=-1,
        max_steps=1e5,
        verbose=True,
        seed=None,
        gradient_accumulation_steps=2,
        report_every=50,
        **kwargs
    ):
        #device, num_gpus = get_device(device=device, num_gpus=num_gpus, local_rank=local_rank)
        device = torch.device("cuda:{}".format(0))
        #gpu_rank = distributed.multi_init(0, 1, "0")
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        
        self.model.to(device)
        
        get_inputs=ExtSumProcessor.get_inputs     
        
        args = Bunch(default_parameters)
        optimizer = model_builder.build_optim(args, self.model, None)
        
        if seed is not None:
            super(ExtractiveSummarizer).set_seed(seed, n_gpu > 0)
        
        train_batch_size = 1,
        
        # multi-gpu training (should be after apex fp16 initialization)
        if num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Distributed training (should be after apex fp16 initialization)
        if local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()
      
        self.model.train()
        start = time.time()
        
        train_data_iterator = train_data_iterator_function()
        accum_loss = 0
        while 1:  
            for step, batch in enumerate(train_data_iterator):
                batch = batch.to(device)

                #batch = tuple(t.to(device) for t in batch if type(t)==torch.Tensor)
                inputs = get_inputs(batch, self.model_name)
                outputs = self.model(**inputs) 
                loss = outputs[0]
                if num_gpus > 1:
                    loss = loss.mean()
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                
                accum_loss += loss.item()
                if step % report_every == 0 and verbose:
                    #tqdm.write(loss)
                    end = time.time()
                    print("loss: {0:.6f}, time: {1:.2f}, step {2:f} out of total {3:f}".format(
                        accum_loss/report_every, end-start, global_step, max_steps))
                    accum_loss = 0
                    start = end
                    
                (loss/loss.numel()).backward()
                
                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    #scheduler.step()
                    self.model.zero_grad()
                    if num_gpus > 1:
                        grads = [p.grad.data for p in self.model.parameters()
                                 if p.requires_grad
                                 and p.grad is not None]
                        distributed.all_reduce_and_rescale_tensors(
                            grads, float(1))
                    
                    global_step += 1

                if max_steps > 0 and global_step > max_steps:
                    break
            if max_steps > 0 and global_step > max_steps:
                break

            # empty cache
            #del [batch]
            torch.cuda.empty_cache()
        return global_step, tr_loss / global_step        

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