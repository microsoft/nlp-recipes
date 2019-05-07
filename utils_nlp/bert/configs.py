from bert_utils import get_device


class GlobalConfig:
    def __init__(self, config_dict={}):
        if "seed" in config_dict:
            self.seed = config_dict["seed"]
        else:
            self.seed = 42

        if "fp16" in config_dict:
            self.fp16 = config_dict["fp16"]
        else:
            self.fp16 = True


class TrainConfig:
    def __init__(self, config_dict={}):
        if 'train_batch_size' in config_dict:
            self.train_batch_size = config_dict['train_batch_size']
        else:
            self.train_batch_size = 32

        if "num_train_epochs" in config_dict:
            self.num_train_epochs = config_dict["num_train_epochs"]
        else:
            self.num_train_epochs = 3

        if "gradient_accumulation_steps" in config_dict:
            self.gradient_accumulation_steps = \
                config_dict["gradient_accumulation_steps"]
        else:
            self.gradient_accumulation_steps = 1

        if "clip_gradient" in config_dict:
            self.clip_gradient = config_dict["clip_gradient"]
        else:
            self.clip_gradient = False

        if "max_gradient_norm" in config_dict:
            self.clip_gradient = config_dict["max_gradient_norm"]
        else:
            self.max_gradient_norm = 1.0


class EvalConfig:
    def __init__(self, config_dict={}):
        if 'eval_batch_size' in config_dict:
            self.eval_batch_size = config_dict['eval_batch_size']
        else:
            self.eval_batch_size = 8


class ModelConfig:
    def __init__(self, config_dict={}):
        self.bert_model = config_dict['bert_model']
        self.max_seq_length = config_dict['max_seq_length']
        self.num_labels = config_dict["num_labels"]
        if "model_type" in config_dict:
            self.model_type = config_dict['model_type']
        else:
            self.model_type = 'sequence'
        if 'do_lower_case' in config_dict:
            self.do_lower_case = config_dict['do_lower_case']
        else:
            self.do_lower_case = True

        if "output_mode" in config_dict:
            self.output_mode = config_dict["output_mode"]
        else:
            self.output_mode = 'classification'


class OptimizerConfig:
    def __init__(self, config_dict={}):
        if 'learning_rate' in config_dict:
            self.learning_rate = config_dict['learning_rate']
        else:
            self.learning_rate = 5e-5

        if "warmup_proportion" in config_dict:
            self.warmup_proportion = config_dict["warmup_proportion"]
        else:
            self.warmup_proportion = 0.1

        if "no_decay_params" in config_dict:
            self.no_decay_params = config_dict["no_decay_params"]
        else:
            # This may depend on the model, so maybe we shouldn't set
            # default values
            self.no_decay_params = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        if "loss_scale" in config_dict:
            self.loss_scale = float(config_dict["loss_scale"])
        else:
            self.loss_scale = 0

        self.num_train_optimization_steps = 1e6


class DeviceConfig:
    def __init__(self, config_dict={}):
        if "no_cuda" in config_dict:
            self.no_cuda = config_dict["no_cuda"]
        else:
            self.no_cuda = True

        if "local_rank" in config_dict:
            self.local_rank = config_dict['local_rank']
        else:
            self.local_rank = -1

        self.device, self.n_gpu = get_device(self.local_rank, self.no_cuda)


class PathConfig:
    def __init__(self, config_dict={}):
        self.data_dir = config_dict['data_dir']
        self.output_dir = config_dict['output_dir']
        if 'cache_dir' in config_dict:
            self.cache_dir = config_dict['cache_dir']
        else:
            self.cache_dir = None
