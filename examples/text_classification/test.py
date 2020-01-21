from utils_nlp.models.mtdnn.configuration_mtdnn import MTDNNConfig
from utils_nlp.models.mtdnn.modeling_mtdnn import MTDNNModel

if __name__ == "__main__":
    config = MTDNNConfig()
    b = MTDNNModel(config)
    print("Config Class: ", b.config_class)
    print("Config: ", b.config)
    print("Embeddings: ", b.embeddings)
    print("Encoding: ", b.encoder)
    print("Pooler: ", b.pooler)
    print("Archive Map: ", b.pretrained_model_archive_map)
    print("Base Model Prefix: ", b.base_model_prefix)
    print("Bert Config: ", b.bert_config)
