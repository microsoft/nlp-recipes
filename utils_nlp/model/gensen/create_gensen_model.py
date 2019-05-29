#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Creates a GenSen model from a MultiSeq2Seq model."""
import os
import pickle

import torch


def create_multiseq2seq_model(trained_model_folder, save_folder, save_name):

    """
    Method that creates a GenSen model from a MultiSeq2Seq model.

    Args:
        trained_model_folder (str): Path to the folder containing a saved model
        save_folder (str): Path to save the encoder
        save_name (str): Name of the model

    Returns: None

    """

    model = torch.load(
        open(os.path.join(trained_model_folder, "best_model.model"), "rb")
    )
    # model.copy() prevents raising the error.
    for item in model.copy().keys():
        if not item.startswith("module.encoder") and not item.startswith(
            "module.src_embedding"
        ):
            model.pop(item)

    for item in model.copy().keys():
        model[item.replace("module.", "")] = model[item]

    for item in model.copy().keys():
        if item.startswith("module."):
            del model[item]

    torch.save(model, os.path.join(save_folder, "%s.model" % save_name))
    # Add 'rb'.
    model_vocab = pickle.load(
        open(os.path.join(trained_model_folder, "src_vocab.pkl"), "rb")
    )
    pickle.dump(
        model_vocab,
        open(os.path.join(save_folder, "%s_vocab.pkl" % save_name), "wb"),
    )
