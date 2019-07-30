import json
import logging
import os

from pytorch_pretrained_bert.tokenization import BasicTokenizer
import torch


from utils_nlp.apps.luis..utterance import Utterance
#from module.util import get_token_span, put_tokens_with_tags
from utils_nlp.models.bert.token_classification import (
    BERTTokenClassifier,
    postprocess_token_labels,
)
from utils_nlp.models.bert.common import Language, Tokenizer

from sklearn.metrics import classification_report
from sklearn_crfsuite.metrics import flat_classification_report, sequence_accuracy_score



logger = logging.getLogger(__name__)

def convert_luis_example_to_tokens_tags(example, basic_tokenizer):
    tags = []
    tokens = []
    if len(example.entities) == 0:
        splitted = basic_tokenizer.tokenize(example.text)
        tags.extend(["O"] * len(splitted))
        tokens.extend(splitted)
    else:
        if example.entities[0].start_pos > 0:
            splitted = basic_tokenizer.tokenize(
                example.text[0 : example.entities[0].start_pos]
            )
            tags.extend(["O"] * len(splitted))
            tokens.extend(splitted)
        for i in range(len(example.entities)):
            splitted = basic_tokenizer.tokenize(
                example.text[
                    example.entities[i].start_pos : example.entities[i].end_pos + 1
                ]
            )
            tags.append("B-" + example.entities[i].entity)
            tags.extend(["I-" + example.entities[i].entity] * (len(splitted) - 1))
            tokens.extend(splitted)
            
            if i + 1 != len(example.entities):
                last_end = example.entities[i].end_pos
                next_begin = example.entities[i + 1].start_pos
                if next_begin >= last_end + 2:
                    splitted = basic_tokenizer.tokenize(
                        example.text[last_end + 1 : next_begin]
                    ) 
                    tags.extend(["O"] * len(splitted))
                    tokens.extend(splitted)

        if example.entities[i].end_pos < len(example.text) - 1:
            splitted = basic_tokenizer.tokenize(
                example.text[example.entities[i].end_pos + 1 : -1]
            ) 
            tags.extend(["O"] * len(splitted))
            tokens.extend(splitted)
    return tokens, tags


def get_token_span(tokens, text):
    """
       tokenize a input utterance, and get the span and POS for each token
    """
    # tokens = tokenization([text])
    #text = text.lower()
    offset, length = 0, 0
    result = []
    point = 0
    for token in tokens:
        if len(token) <= 0:
            continue
        ## for bert tokenizer
        if token.startswith("##"):
            #print(token[2:])
            found_start = text.find(token[2:], point)
            if found_start == -1:
                raise Exception("Subword not found, possile tokenization error!")
            found_end = found_start + len(token) - 2
        else:
            found_start = text.find(token, point)
            if found_start == -1:
                raise Exception("Token not found, possile tokenization error!")
            found_end = found_start + len(token)
        result.append(
            {
                "text": token,
                "startPos": found_start,
                "endPos": found_end - 1,  # , "POS": pos_text[i][1]
            }
        )
        point = found_end
    return result


def make_trailing_tags_explicit(tags, trailing_piece_tag="X"):
    """['O', 'B-ContactName', 'X', 'X', 'O']
     =>['O', 'B-ContactName', 'B-ContactName', 'B-ContactName', 'O'] 
    """
    new_tags = []
    #if tags[0] == trailing_piece_tag:
    #    raise Exception(
    #        "the start of a tag sequence should not be a trailing piece tag"
    #    )
    last_tag = tags[0]
    new_tags.append(last_tag)
    for i in range(1, len(tags)):
        if tags[i] == trailing_piece_tag:
            new_tags.append(last_tag)
        else:
            last_tag = tags[i]
            new_tags.append(last_tag)
    return new_tags


def remove_ib_from_tag(tags):
    new_tags = []
    for i in tags:
        splitted = i.split("-")
        if len(splitted) >= 2:
            new_tags.append(splitted[1])
        else:
            new_tags.append(splitted[0])
    return new_tags

def put_tokens_with_tags(text, tokens, tags):
    """
        merge the adjacent tokens to a single entity in the returned entity list
        if these adjacent tokens have the same tag
    tokens
    [{'endPos': 3, 'startPos': 0, 'token': 'Call'},
      {'endPos': 8, 'startPos': 6, 'token': 'Amy'},
      {'endPos': 11, 'startPos': 10, 'token': 'Hu'}]
    taggs
    ['O', 'Contact.Name', 'Contact.Name']
    return result
    [{'endPos': 11, 'startPos': 6, 'token': 'Amy Hu'}]
   """

    result = []
    i = 0
    start = 0
    while i < len(tags):
        if tags[i]["tag"] == "O":
            i += 1
            start = i
            continue
        if i < len(tags) - 1 and tags[i]["tag"] == tags[i + 1]["tag"]:
            i += 1
        else:
            result.append(
                Entity(
                    {
                        "entity": tags[start]["tag"],
                        "confidence": tags[start]["confidence"],
                        "startPos": tokens[start]["startPos"],
                        "endPos": tokens[i]["endPos"],
                        "value": text[tokens[start]["startPos"]:tokens[i]["endPos"]+1]
                    }
                )
            )
            i += 1
            start = i
    return result


# set random seeds
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

class BERTEntityExtractor:
    def __init__(
        self,
        language=Language.ENGLISHCASED,
        num_train_epochs=10,
        do_lower_case=False,
        max_seq_length=100,
        batch_size=16,
        learning_rate=3e-5,
        cache_dir='./temp',
        num_gpus=None
    ):
        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)
        if not os.path.isdir(cache_dir):
            raise Exception(
                "Please check permission if you can create or access the cache dir {0}".format(
                    cache_dir
                )
            )
        self.cache_dir = cache_dir
        self.language = language
        self.do_lower_case = do_lower_case
        # training configurations
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.num_train_epochs = num_train_epochs
        # optimizer configuration
        self.learning_rate = learning_rate
        self.num_gpus = num_gpus
        # for serialization
        self.saved_model = None
        self.label_map = None
        self.saved_luis_model = None
        
        # tokenizer
        self.tokenizer = Tokenizer(
            language=self.language, to_lower=self.do_lower_case, cache_dir=self.cache_dir
        )

    def save(self, extractor_file, model_file=None, label_map_file=None):
        torch.save(self, extractor_file)
        if model_file:
            torch.save(self.saved_model, model_file)
        if label_map_file:
            torch.save(self.label_map, label_map_file)

    def load(self, model_file, label_map_file):
        if torch.cuda.is_available():
            self.saved_model = torch.load(model_file)
            self.label_map = torch.load(label_map_file)
        else:
            self.saved_model = torch.load(model_file, location='cpu')
            self.label_map = torch.load(label_map_file, location='cpu')

    def prepare_training_data(self, luis_model_file):
        # load utterance
        with open(luis_model_file, "r") as luis_fd:
            luis_model = json.load(luis_fd)
            self.saved_luis_model = luis_model

        utterances = []
        for utterance_obj in luis_model["utterances"]:
            utterances.append(Utterance(utterance_obj))

        tokens_list = []
        tags_list = []
        bt = BasicTokenizer()
        for i in utterances:
            logger.debug(i.text)
            tokens, tags = split_to_tokens_tags(i, bt)
            logger.debug(tokens)
            logger.debug(tags)
            tokens_list.append(tokens)
            tags_list.append(tags)
            logger.debug(tokens)
            logger.debug(tags)
        return utterances, tokens_list, tags_list

    def train(self, luis_model_file):
        utterances, tokens_list, tags_list = self.prepare_training_data(luis_model_file)
        train_text = []
        train_labels = []
        for i in range(len(utterances)):
            train_text.append(tokens_list[i])
            train_labels.append(tags_list[i])

        tags_set = set()
        for i in tags_list:
            for j in i:
                tags_set.add(j)
        label_list = list(tags_set)
        label_map = {}
        
        label_map = {label: i for i, label in enumerate(label_list)}
        label_map["X"] = len(label_list)
        logger.info(label_map)


        train_token_ids, train_input_mask, train_trailing_token_mask, train_label_ids = self.tokenizer.tokenize_ner(
            text=train_text,
            label_map=label_map,
            max_len=self.max_seq_length,
            labels=train_labels,
            trailing_piece_tag="X",
        )
        token_classifier = BERTTokenClassifier(
            language=self.language, num_labels=len(label_map.keys()), cache_dir=self.cache_dir
        )
        token_classifier.fit(
            token_ids=train_token_ids,
            input_mask=train_input_mask,
            labels=train_label_ids,
            num_epochs=self.num_train_epochs,
            batch_size=self.batch_size,
            num_gpus=self.num_gpus,
            learning_rate=self.learning_rate,
        )

        pred_label_ids = token_classifier.predict(
            token_ids=train_token_ids,
            input_mask=train_input_mask,
            labels=train_label_ids,
            batch_size=self.batch_size,
        )
        # get training performance
        test_input_mask = train_input_mask
        test_label_ids = train_label_ids
        ## post processing 
        pred_tags_no_padding = postprocess_token_labels(
            pred_label_ids, test_input_mask, label_map
        )
        true_tags_no_padding = postprocess_token_labels(
            test_label_ids, test_input_mask, label_map
        )
        report = flat_classification_report(
            y_pred=pred_tags_no_padding, y_true=true_tags_no_padding
        )
        logger.info(report)
        logger.info(
            sequence_accuracy_score(
                y_pred=pred_tags_no_padding, y_true=true_tags_no_padding
            )
        )
        # save for serialization
        self.saved_model = token_classifier
        self.label_map = label_map

    def copy_for_predict(self, external_model):
        self.tokenizer = external_model.tokenizer 
        self.saved_model = external_model.saved_model
        self.label_map = external_model.label_map
        self.batch_size = external_model.batch_size
        self.saved_luis_model = external_model.saved_luis_model

    def predict(self, text, progress=True):
        # save the position of each token
        # TODO: disable progress bar in tokenizer.tokenize()
        raw_tokens = self.tokenizer.tokenize(
                [text],
                #progress=progress,
                )[0]
        logger.debug("test example:{}".format(text))
        logger.debug("raw tokens: {}".format(raw_tokens))
        tokens_dict = get_token_span(raw_tokens, text)
        logger.debug("splitted token:{}".format(tokens_dict))

        splitted = text.split()
        # use bert token classifier to predict
        test_token_ids, test_input_mask, test_trailing_token_mask, test_label_ids = self.tokenizer.tokenize_ner(
            text=[splitted],
            label_map=self.label_map,
            max_len=self.max_seq_length,
            labels=[["O"] * len(splitted)],
            trailing_piece_tag="X",
        )
        if torch.cuda.is_available():
            num_gpus = None # use all gpus
        else:
            num_gpus = 0

        # TODO: disable progress bar in model.predict()
        # TODO: other debug messages output here too
        prediction = self.saved_model.predict(
            token_ids=test_token_ids,
            input_mask=test_input_mask,
            labels=test_label_ids,
            batch_size=self.batch_size,
            probabilities=True,
            num_gpus = num_gpus,
            #progress=progress,
        )
        pred_label_ids = prediction.classes
        pred_label_prob = prediction.probabilities[0]
        pred_tags_no_padding = postprocess_token_labels(
            pred_label_ids, test_input_mask, self.label_map
        )
        logger.debug("predcted tags: {}".format(pred_tags_no_padding))

        # post processing
        pred_tags_without_trailing = make_trailing_tags_explicit(
            pred_tags_no_padding[0]
        )

        pred_tags_without_trailing = remove_ib_from_tag(
            # pred_tags_no_padding[0]
            pred_tags_without_trailing
        )
        logger.debug("merged tags: {}".format(pred_tags_without_trailing))
        tags_dict = []
        for i in range(len(pred_tags_without_trailing)):
            tags_dict.append(
                {"tag": pred_tags_without_trailing[i], "confidence": pred_label_prob[i]}
            )
        simple_entities = put_tokens_with_tags(text, tokens_dict, tags_dict)
        logger.debug("simple entities: {}".format(simple_entities))
        return simple_entities
