from utils_nlp.models.mtdnn.configuration_mtdnn import MTDNNConfig
from utils_nlp.models.mtdnn.modeling_mtdnn import MTDNNModel

if __name__ == "__main__":
    config = MTDNNConfig()
    b = MTDNNModel(config)
    print(b.config_class)
    print(b.config)
    print(b.embeddings)
    print(b.encoder)
    print(b.pooler)
    print(b.pretrained_model_archive_map)
    print(b.base_model_prefix)
    print(b.bert_config)
