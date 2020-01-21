from utils_nlp.models.mtdnn.common.types import EncoderModelType
from utils_nlp.models.mtdnn.configuration_mtdnn import MTDNNConfig
from utils_nlp.models.mtdnn.modeling_mtdnn import MTDNNModel

if __name__ == "__main__":
    config = MTDNNConfig()
    b = MTDNNModel(config)
    print("Network: ", b.network)
    print("Config Class: ", b.config_class)
    print("Config: ", b.config)
    print("Pooler: ", b.pooler)

    if config.encoder_type == EncoderModelType.BERT:
        print("Encoding: ", b.encoder)
        print("Embeddings: ", b.embeddings)
        print("Bert Config: ", b.bert_config)

    print("Archive Map: ", b.pretrained_model_archive_map)
    print("Base Model Prefix: ", b.base_model_prefix)
