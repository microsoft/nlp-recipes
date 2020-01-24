# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.

from utils_nlp.models.mtdnn.tasks.config import TaskDefs


class MTDNNDataPreprocess:
    def __init__(
        self,
        task_defs: TaskDefs,
        batch_size: int,
        data_dir: str = "data/canonical_data/bert_uncased_lower",
        train_datasets_list: str = ["mnli"],
        test_datasets_list: str = ["mnli_mismatched,mnli_matched"],
        glue_format: bool = False,
        data_sort: bool = False,
    ):
        assert len(train_datasets_list) >= 1, "Train dataset list cannot be empty"
        assert len(test_datasets_list) >= 1, "Test dataset list cannot be empty"

        # Initialize definition maps
        self.task_defs = task_defs
        self.train_datasets = train_datasets_list
        self.test_datasets = test_datasets_list
        self.data_dir = data_dir
        self.glue_format = glue_format
        self.data_sort = data_sort
        self.batch_size = batch_size
        self.tasks = {}
        self.tasks_class = {}
        self.nclass_list = []
        self.decoder_opts = []
        self.task_types = []
        self.dropout_list = []
        self.loss_types = []
        self.kd_loss_types = []

    def process_train_datasets(self):

        for dataset in self.train_datasets:
            prefix = dataset.split("_")[0]
            if prefix in self.tasks:
                continue
            assert prefix in self.task_defs.n_class_map
            assert prefix in self.task_defs.data_type_map
            data_type = self.task_defs.data_type_map[prefix]
            nclass = self.task_defs.n_class_map[prefix]
            task_id = len(tasks)
            if args.mtl_opt > 0:
                task_id = tasks_class[nclass] if nclass in tasks_class else len(tasks_class)

            task_type = self.task_defs.task_type_map[prefix]

            dopt = self.generate_decoder_opt(
                self.task_defs.enable_san_map[prefix], opt["answer_opt"]
            )
            if task_id < len(decoder_opts):
                decoder_opts[task_id] = min(decoder_opts[task_id], dopt)
            else:
                decoder_opts.append(dopt)
            task_types.append(task_type)
            loss_types.append(self.task_defs.loss_map[prefix])
            kd_loss_types.append(self.task_defs.kd_loss_map[prefix])

            if prefix not in tasks:
                tasks[prefix] = len(tasks)
                if args.mtl_opt < 1:
                    nclass_list.append(nclass)

            if nclass not in tasks_class:
                tasks_class[nclass] = len(tasks_class)
                if args.mtl_opt > 0:
                    nclass_list.append(nclass)

            dropout_p = self.task_defs.dropout_p_map.get(prefix, args.dropout_p)
            dropout_list.append(dropout_p)

            train_path = os.path.join(data_dir, "{}_train.json".format(dataset))
            logger.info("Loading {} as task {}".format(train_path, task_id))
            train_data_set = SingleTaskDataset(
                train_path,
                True,
                maxlen=args.max_seq_len,
                task_id=task_id,
                task_type=task_type,
                data_type=data_type,
            )
            train_datasets.append(train_data_set)

    def generate_decoder_opt(self, enable_san, max_opt):
        opt_v = 0
        if enable_san and max_opt < 3:
            opt_v = max_opt
        return opt_v
