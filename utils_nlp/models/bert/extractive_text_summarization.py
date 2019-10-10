from pytorch_pretrained_bert import BertConfig

from bertsum.models.model_builder import Summarizer
from bertsum.models import  model_builder, data_loader
from bertsum.others.logging import logger, init_logger
from bertsum.train import model_flags
from bertsum.models.trainer import build_trainer
from bertsum.prepro.data_builder import BertData

from cached_property import cached_property
import torch
import random
from bertsum.prepro.data_builder import greedy_selection, combination_selection
import gc
from multiprocessing import Pool


class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)

default_parameters = {"accum_count": 1, "batch_size": 3000, "beta1": 0.9, "beta2": 0.999, "block_trigram": True, "decay_method": "noam", "dropout": 0.1, "encoder": "baseline", "ff_size": 512, "gpu_ranks": "0123", "heads": 4, "hidden_size": 128, "inter_layers": 2, "lr": 0.002, "max_grad_norm": 0, "max_nsents": 100, "max_src_ntokens": 200, "min_nsents": 3, "min_src_ntokens": 10, "optim": "adam", "oracle_mode": "combination", "param_init": 0.0, "param_init_glorot": True, "recall_eval": False, "report_every": 50, "report_rouge": True, "rnn_size": 512, "save_checkpoint_steps": 500, "seed": 666, "temp_dir": "./temp", "test_all": False, "test_from": "", "train_from": "", "use_interval": True, "visible_gpus": "0", "warmup_steps": 10000, "world_size": 1}

default_preprocessing_parameters = {"max_nsents": 100, "max_src_ntokens": 200, "min_nsents": 3, "min_src_ntokens": 10, "use_interval": True}

def preprocess(client, source, target):
    pre_source = tokenize_to_list(source, client)
    pre_target = tokenize_to_list(target, client)
    return bertify(pre_source, pre_target)

def tokenize_to_list(client, input_text):
    annotation = client.annotate(input_text)
    sentences = annotation.sentence
    tokens_list = []
    for sentence in sentences:
        tokens = []
        for token in sentence.token:
            tokens.append(token.originalText)
        tokens_list.append(tokens)
    return tokens_list

def bertify(bertdata, source, target=None, oracle_mode='combination', selection=3):
    if target:
        oracle_ids = combination_selection(source, target, selection)
        b_data = bertdata.preprocess(source, target, oracle_ids)
    else:
        b_data = bertdata.preprocess(source, None, None)
    if b_data is None:
        return None
    indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt = b_data
    b_data_dict = {"src": indexed_tokens, "labels": labels, "segs": segments_ids, 'clss': cls_ids,
                   'src_txt': src_txt, "tgt_txt": tgt_txt}
    return b_data_dict
    #gc.collect()


def bertsum_formatting(n_cpus, bertdata, oracle_mode, jobs, output_file):
    params = []
    for i in jobs:
        params.append((oracle_mode, bertdata, i))
    pool = Pool(n_cpus)
    bert_data = pool.map(modified_format_to_bert, params, int(len(params)/n_cpus))
    pool.close()
    pool.join()
    filtered_bert_data = []
    for i in bert_data:
        if i is not None:
            filtered_bert_data.append(i)
    torch.save(filtered_bert_data, output_file)

def modified_format_to_bert(param):
    oracle_mode, bert, data = param
    #return data
    source, tgt = data['src'], data['tgt']
    if (oracle_mode == 'greedy'):
        oracle_ids = greedy_selection(source, tgt, 3)
    elif (oracle_mode == 'combination'):
        oracle_ids = combination_selection(source, tgt, 3)
    b_data = bert.preprocess(source, tgt, oracle_ids)
    if (b_data is None):
        return None
    indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt = b_data
    b_data_dict = {"src": indexed_tokens, "labels": labels, "segs": segments_ids, 'clss': cls_ids,
                   'src_txt': src_txt, "tgt_txt": tgt_txt}
    return b_data_dict
    gc.collect()


class BertSumExtractiveSummarizer:
    """BERT-based Extractive Summarization --BertSum"""
    
    
    def __init__(self, language="english",  
                  mode = "train",
                  encoder="baseline",
                  model_path = "./models/baseline",
                  log_file = "./logs/baseline",
                  temp_dir = './temp',
                  bert_config_path="./bert_config_uncased_base.json",
                  device_id=1,
                  work_size=1,
                  gpu_ranks="1"
                  ):
        """Initializes the classifier and the underlying pretrained model.
        Args:
            language (Language, optional): The pretrained model's language.
                                           Defaults to Language.ENGLISH.
            num_labels (int, optional): The number of unique labels in the
                training data. Defaults to 2.
            cache_dir (str, optional): Location of BERT's cache directory.
                Defaults to ".".
        """
        def __map_gpu_ranks(gpu_ranks):
            gpu_ranks_list=gpu_ranks.split(',')
            print(gpu_ranks_list)
            gpu_ranks_map = {}
            for i, rank in enumerate(gpu_ranks_list):
                gpu_ranks_map[int(rank)]=i
            return gpu_ranks_map
    
        
        # copy all the arguments from the input argument
        self.args = Bunch(default_parameters)
        self.args.seed = 42
        self.args.mode = mode
        self.args.encoder = encoder
        self.args.model_path = model_path
        self.args.log_file = log_file
        self.args.temp_dir = temp_dir
        self.args.bert_config_path=bert_config_path
        
        self.args.gpu_ranks = gpu_ranks
        self.args.gpu_ranks_map = __map_gpu_ranks(self.args.gpu_ranks) 
        self.args.world_size = len(self.args.gpu_ranks_map.keys())
        print(self.args.gpu_ranks_map)
    

        self.has_cuda = self.cuda
        init_logger(self.args.log_file) 
        torch.manual_seed(self.args.seed)
        random.seed(self.args.seed)
        torch.backends.cudnn.deterministic = True
        # placeholder for the model
        self.model = None 
        
    @cached_property
    def cuda(self):
        """ cache the output of torch.cuda.is_available() """

        self.has_cuda = torch.cuda.is_available()
        return self.has_cuda
        
    
    def fit(self, device_id, train_file_list, train_steps=5000, train_from='', batch_size=3000, 
               warmup_proportion=0.2, decay_method='noam', lr=0.002,accum_count=2):

        if self.args.gpu_ranks_map[device_id] != 0:
            logger.disabled = True
        if device_id not in list(self.args.gpu_ranks_map.keys()): 
             raise Exception("need to use device id that's in the gpu ranks")
        device = None
        if device_id >= 0:
            #torch.cuda.set_device(device_id)
            torch.cuda.manual_seed(self.args.seed)
            device = device_id #torch.device("cuda:{}".format(device_id))  
        
        self.args.decay_method=decay_method
        self.args.lr=lr
        self.args.train_from = train_from
        self.args.batch_size = batch_size
        self.args.warmup_steps = int(warmup_proportion*train_steps)
        self.args.accum_count= accum_count
        print(self.args.__dict__)
        
        self.model = Summarizer(self.args, device, load_pretrained_bert=True)
    
    
        if train_from != '':
            checkpoint = torch.load(train_from,
                                    map_location=lambda storage, loc: storage)
            opt = vars(checkpoint['opt'])
            for k in opt.keys():
                if (k in model_flags):
                    setattr(self.args, k, opt[k])
            self.model.load_cp(checkpoint)
            optim = model_builder.build_optim(self.args, self.model, checkpoint)
        else:
            optim = model_builder.build_optim(self.args, self.model, None)

        
        def get_dataset(file_list):
            random.shuffle(file_list)
            for file in file_list:
                yield torch.load(file)
        

        def train_iter_fct():
            return data_loader.Dataloader(self.args, get_dataset(train_file_list), batch_size, device,
                                             shuffle=True, is_test=True)
        


        trainer = build_trainer(self.args, device_id, self.model, optim)
        trainer.train(train_iter_fct, train_steps)

    def predict(self, device_id, data_iter, sentence_seperator='', test_from='', cal_lead=False):
        ## until a fix comes in
        #if self.args.world_size=1 or len(self.args.gpu_ranks.split(",")==1):
        #    device_id = 0
            
        device = None
        if device_id >= 0:
            torch.cuda.set_device(device_id)
            torch.cuda.manual_seed(self.args.seed)
            device = torch.device("cuda:{}".format(device_id)) 
            
        if self.model is None and test_from == '':
            raise Exception("Need to train or specify the model for testing")
        if test_from != '':
            checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
            opt = vars(checkpoint['opt'])
            for k in opt.keys():
                if (k in model_flags):
                    setattr(self.args, k, opt[k])
                    
            config = BertConfig.from_json_file(self.args.bert_config_path)
            self.model = Summarizer(self.args, device, load_pretrained_bert=False, bert_config=config)
            self.model.load_cp(checkpoint)
        else:
            #model = self.model
            self.model.eval()
        self.model.eval()

        trainer = build_trainer(self.args, device_id, self.model, None)
        return trainer.predict(data_iter, sentence_seperator, cal_lead)
        
