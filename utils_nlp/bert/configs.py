class BERTFineTuneConfig:
    """
    Configurations for fine tuning pre-trained BERT models.
    """

    def __init__(self, config_dict):
        """
        Initializes a BERTFineTuneConfig object.

        Args:
            config_dict (dict): A nested dictionary containing three key,
                value pairs.
                "ModelConfig": model configuration dictionary:
                    "bert_model": str, name of the bert pre-trained model.
                        Accepted values are:
                        "bert-base-uncased"
                        "bert-large-uncased"
                        "bert-base-cased"
                        "bert-large-cased"
                        "bert-base-multilingual-uncased"
                        "bert-base-multilingual-cased"
                        "bert-base-chinese"
                    "max_seq_length": int, optional. Maximum length of token
                        sequence. Default value is 512.
                    "do_lower_case": bool, optional. Whether to convert
                        capital letters to lower case during tokenization.
                        Default value is True.
                "TrainConfig" (optional): training configuration dictionary:
                    "batch_size": int, optional. Default value is 32.
                    "num_train_epochs": int, optional. Default value is 3.
                "OptimizerConfig" (optional): optimizer configuration
                    dictionary:
                    "optimizer_name": str, optional. Name of the optimizer
                        to use. Accepted values are "BertAdam", "Adam".
                        Default value is "BertAdam"
                    "learning_rate": float, optional, default value is 5e-05,
                    "no_decay_params": list of strings, optional. Names of
                        parameters to apply weight decay on. Default value
                        is [].
                    "params_weight_decay": float, optional. Parameter weight
                        decay rate. Default value is 0.01.
                    "clip_gradient": bool, optional. Whether to perform
                        gradient clipping. Default value is False.
                    "max_gradient_norm": float, optional. Maximum gradient
                        norm to apply gradient clipping on. Default value is
                        1.0.
        """

        train_config_dict = {}
        optimizer_config_dict = {}

        if "TrainConfig" in config_dict:
            train_config_dict = config_dict["TrainConfig"]

        if "OptimizerConfig" in config_dict:
            optimizer_config_dict = config_dict["OptimizerConfig"]

        if "ModelConfig" in config_dict:
            model_config_dict = config_dict["ModelConfig"]
        else:
            raise Exception("The 'ModelConfig' field can not be empty.")

        self._configure_train_settings(train_config_dict)
        self._configure_model_settings(model_config_dict)
        self._configure_optimizer_settings(optimizer_config_dict)

    def _configure_train_settings(self, config_dict):
        if "batch_size" in config_dict:
            self.batch_size = config_dict["batch_size"]
        else:
            self.batch_size = 32
            print(
                "batch_size is set to default value: {}.".format(
                    self.batch_size
                )
            )

        if "num_train_epochs" in config_dict:
            self.num_train_epochs = config_dict["num_train_epochs"]
        else:
            self.num_train_epochs = 3
            print(
                "num_train_epochs is set to default value: {}.".format(
                    self.num_train_epochs
                )
            )

    def _configure_model_settings(self, config_dict):
        self.bert_model = config_dict["bert_model"]

        if "max_seq_length" in config_dict:
            self.max_seq_length = config_dict["max_seq_length"]
        else:
            self.max_seq_length = 512
            print(
                "max_seq_length is set to default value: {}.".format(
                    self.max_seq_length
                )
            )

        if "do_lower_case" in config_dict:
            self.do_lower_case = config_dict["do_lower_case"]
        else:
            self.do_lower_case = True
            print(
                "do_lower_case is set to default value: {}.".format(
                    self.do_lower_case
                )
            )

    def _configure_optimizer_settings(self, config_dict):
        if "optimizer_name" in config_dict:
            self.optimizer_name = config_dict["optimizer_name"]
        else:
            self.optimizer_name = "BertAdam"
            print(
                "optimizer_name is set to default value: {}.".format(
                    self.optimizer_name
                )
            )

        if "learning_rate" in config_dict:
            self.learning_rate = config_dict["learning_rate"]
        else:
            self.learning_rate = 5e-5
            print(
                "learning_rate is set to default value: {}.".format(
                    self.learning_rate
                )
            )

        if "no_decay_params" in config_dict:
            self.no_decay_params = config_dict["no_decay_params"]
        else:
            self.no_decay_params = []
            print(
                "no_decay_params is set to default value: {}.".format(
                    self.no_decay_params
                )
            )

        if "params_weight_decay" in config_dict:
            self.params_weight_decay = config_dict["params_weight_decay"]
        else:
            self.params_weight_decay = 0.01
            print(
                "Default params_weight_decay, {}, is used".format(
                    self.params_weight_decay
                )
            )

        if "clip_gradient" in config_dict:
            self.clip_gradient = config_dict["clip_gradient"]
        else:
            self.clip_gradient = False
            print(
                "clip_gradient is set to default value: {}.".format(
                    self.clip_gradient
                )
            )

        if "max_gradient_norm" in config_dict:
            self.max_gradient_norm = config_dict["max_gradient_norm"]
        else:
            self.max_gradient_norm = 1.0
            print(
                "max_gradient_norm is set to default value: {}.".format(
                    self.max_gradient_norm
                )
            )

    def __str__(self):
        sb = []
        for key in self.__dict__:
            sb.append(
                "{key}={value}".format(key=key, value=self.__dict__[key])
            )

        return "\n".join(sb)
