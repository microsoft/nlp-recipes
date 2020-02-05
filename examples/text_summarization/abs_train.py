#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')


# In[2]:


#get_ipython().run_line_magic('autoreload', '2')


# In[3]:


import os
import shutil
import sys
from tempfile import TemporaryDirectory
import torch

nlp_path = os.path.abspath("../../")
if nlp_path not in sys.path:
    sys.path.insert(0, nlp_path)

from utils_nlp.common.pytorch_utils import get_device
from utils_nlp.dataset.cnndm import CNNDMBertSumProcessedData, CNNDMSummarizationDataset
from utils_nlp.eval.evaluate_summarization import get_rouge
from utils_nlp.models.transformers.extractive_summarization import (
    ExtractiveSummarizer,
    ExtSumProcessedData,
    ExtSumProcessor,
)

import numpy as np
import pandas as pd
import scrapbook as sb


# In[4]:


from utils_nlp.dataset.cnndm import CNNDMSummarizationDataset


# In[5]:


DATA_PATH = '/tmp/tmp38t9l0ek' #TemporaryDirectory().name
QUICK_RUN = False
# the data path used to save the downloaded data file
#DATA_PATH = TemporaryDirectory().name
# The number of lines at the head of data file used for preprocessing. -1 means all the lines.
TOP_N = 4
CHUNK_SIZE=200
if not QUICK_RUN:
    TOP_N = -1
    CHUNK_SIZE = 2000


# In[6]:


#train_dataset, test_dataset = CNNDMSummarizationDataset(top_n=TOP_N, local_cache_path=DATA_PATH, tokenize_sentence=False)


# In[7]:


from torch.utils.data import Dataset
class SummarizationNonIterableDataset(Dataset):
    def __init__(self, source, target=None):
        self.source = source
        self.target = target
    def __len__(self):
        return len(self.source)
    def __getitem__(self, idx):
        return self.source[idx], self.target[idx]


# In[8]:


#data = list(test_dataset.get_source()), list(test_dataset.get_target())
#test_sum_dataset = SummarizationNonIterableDataset(data[0], data[1])


# In[9]:


#data = list(train_dataset.get_source()), list(train_dataset.get_target())
#train_sum_dataset = SummarizationNonIterableDataset(data[0], data[1])


# In[10]:


from utils_nlp.models.transformers.abssum import AbsSumProcessor 


# In[11]:


#torch.save(test_sum_dataset, "test_sum_dataset.pt")


# In[12]:


#torch.save(train_sum_dataset, "train_sum_dataset.pt")


# In[13]:


train_sum_dataset = torch.load("train_sum_dataset.pt")
test_sum_dataset = torch.load("test_sum_dataset.pt")


# In[14]:


processor = AbsSumProcessor()


# In[15]:


#processor.collate(train_sum_dataset, 64, "cuda")


# In[16]:


from utils_nlp.common.pytorch_utils import get_device
device, num_gpus = get_device(num_gpus=1, local_rank=-1)


# In[17]:


from utils_nlp.models.transformers.abssum import AbsSum


# In[18]:


summarize = AbsSum(processor)


# In[ ]:


from torch.utils.data import SequentialSampler, RandomSampler, DataLoader
def build_data_iterator(collate, dataset, batch_size=16, device='cuda'):

    sampler = RandomSampler(dataset)

    def collate_fn(data):
        return collate(data, block_size=512, device=device)

    iterator = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate_fn,)

    return iterator
# batch_size is the number of tokens in a batch
#train_dataloader = get_dataloader(train_dataset.get_stream(), is_labeled=True, batch_size=batch_size)
#train_dataloader = build_data_iterator(processor.collate, train_sum_dataset, batch_size=4, device='cuda:0')


# In[ ]:


#for i in train_dataloader:
#    print(i.src[0])
#    break


# In[19]:


#list(summarize.model.named_parameters())


# In[21]:


summarize.fit(train_sum_dataset, batch_size=4, fp16=False, max_steps=1000, num_gpus=1, verbose=True)
torch.save(summarize, 'abs_{}.pt'.format(max_step))

# In[ ]:





