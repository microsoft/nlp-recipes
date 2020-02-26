from utils_nlp.models.mtdnn.common.types import EncoderModelType
from utils_nlp.models.mtdnn.configuration_mtdnn import MTDNNConfig
from utils_nlp.models.mtdnn.modeling_mtdnn import MTDNNModel
from utils_nlp.models.mtdnn.process_mtdnn import MTDNNDataProcess, MTDNNPipelineProcess
from utils_nlp.models.mtdnn.tasks.config import MTDNNTaskDefs

if __name__ == "__main__":
    config = MTDNNConfig()
    #config.log_per_updates = 10
    print(config)
    # Define task parameters
    tasks_params = {
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

    # Define the tasks
    task_defs = MTDNNTaskDefs(tasks_params)

    # Make the Data Preprocess step and update the config with training data updates
    # data_processor = MTDNNDataProcess(
    #     config=config,
    #     task_defs=task_defs,
    #     batch_size=8,
    #     data_dir="/home/useradmin/sources/mt-dnn/data/canonical_data/bert_uncased_lower",
    #     train_datasets_list=["mnli"],
    #     test_datasets_list=["mnli_mismatched, mnli_matched"],
    # )

    # Make the Data Preprocess step and update the config with training data updates
    data_processor = MTDNNDataProcess(
        config=config,
        task_defs=task_defs,
        batch_size=16,
        data_dir="D:/experiments/codes/mt-dnn-private/data/mt_dnn_uncased_lower",
        train_datasets_list=["mnli"],
        test_datasets_list=["mnli_mismatched", "mnli_matched"],
    )

    multitask_train_dataloader = data_processor.get_train_dataloader()
    dev_dataloaders_list = data_processor.get_dev_dataloaders()
    test_dataloaders_list = data_processor.get_test_dataloaders()

    # Update config with data preprocess params
    config = data_processor.update_config(config)
    
    # Update steps
    num_all_batches = config.epochs * len(multitask_train_dataloader) // config.grad_accumulation_step

    # Instantiate model
    #import pdb; pdb.set_trace()
    model = MTDNNModel(config, pretrained_model_name= "bert-base-uncased", num_train_step=num_all_batches,)
    print("Network: ", model.network)
    #import pdb; pdb.set_trace()
    # Create a process pipeline for training and inference
    pipeline_process = MTDNNPipelineProcess(
        model=model,
        config=config,
        task_defs=task_defs,
        multitask_train_dataloader=multitask_train_dataloader,
        dev_dataloaders_list=dev_dataloaders_list,
        test_dataloaders_list=test_dataloaders_list,
    )
    # Fit training data to model
    pipeline_process.fit(1)
    pipeline_process.predict(1)

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
