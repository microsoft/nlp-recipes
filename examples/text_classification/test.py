from utils_nlp.models.mtdnn.common.types import EncoderModelType
from utils_nlp.models.mtdnn.configuration_mtdnn import MTDNNConfig
from utils_nlp.models.mtdnn.modeling_mtdnn import MTDNNModel
from utils_nlp.models.mtdnn.process_mtdnn import MTDNNDataPreprocess
from utils_nlp.models.mtdnn.tasks.config import TaskDefs

if __name__ == "__main__":
    config = MTDNNConfig()

    # Define task parameters
    cola_opts = {
        # "cola": {
        #     "data_format": "PremiseOnly",
        #     "encoder_type": "BERT",
        #     "dropout_p": 0.05,
        #     "enable_san": False,
        #     "metric_meta": ["ACC", "MCC"],
        #     "loss": "CeCriterion",
        #     "kd_loss": "MseCriterion",
        #     "n_class": 2,
        #     "task_type": "Classification",
        # },
        "mnli": {
            "data_format": "PremiseAndOneHypothesis",
            "encoder_type": "BERT",
            "dropout_p": 0.3,
            "enable_san": True,
            "labels": ["contradiction", "neutral", "entailment"],
            "metric_meta": ["ACC"],
            "loss": "CeCriterion",
            "kd_loss": "MseCriterion",
            "n_class": 3,
            "split_names": [
                "train",
                "matched_dev",
                "mismatched_dev",
                "matched_test",
                "mismatched_test",
            ],
            "task_type": "Classification",
        },
    }
    task_defs = TaskDefs(cola_opts)

    # Make the Data Preprocess step and update the config with training data updates
    processor = MTDNNDataPreprocess(
        config=config,
        task_defs=task_defs,
        batch_size=8,
        data_dir="/home/useradmin/sources/mt-dnn/data/canonical_data/bert_uncased_lower",
        train_datasets_list=["mnli"],
        test_datasets_list=["mnli_mismatched, mnli_matched"],
    )

    # Update config with data preprocess params
    config = processor.update_config(config)
    model = MTDNNModel(config)
    print("Network: ", model.network)
    # print("Config Class: ", b.config_class)
    # print("Config: ", b.config)
    # print("Pooler: ", b.pooler)
    # print("Encoding: ", b.encoder)
    # print("Embeddings: ", b.embeddings)

    # if config.encoder_type == EncoderModelType.BERT:
    #     print("Bert Config: ", b.bert_config)

    # print("Archive Map: ", b.pretrained_model_archive_map)
    # print("Base Model Prefix: ", b.base_model_prefix)

    # Training and inference
    # model.train(...)
    # model.eval(...)
