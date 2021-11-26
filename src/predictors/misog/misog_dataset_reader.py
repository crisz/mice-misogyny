from typing import Dict, List, Optional
import logging

from allennlp.data import Tokenizer
from overrides import overrides
from nltk.tree import Tree


from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field, IndexField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.common.checks import ConfigurationError

from pathlib import Path
from itertools import chain
import os.path as osp
import tarfile
from tqdm import tqdm as tqdm
import numpy as np
import pandas as pd
import math
from sklearn.datasets import fetch_20newsgroups

from src.predictors.predictor_utils import clean_text

logger = logging.getLogger(__name__)

TRAIN_VAL_SPLIT_RATIO = 0.7

DATASET_PATH_BASE = Path('.') / 'data'
DEV_PATH = DATASET_PATH_BASE / 'misogyny_EN' / 'miso_dev.tsv'
TRAIN_PATH = DATASET_PATH_BASE / 'misogyny_EN' / 'miso_train.tsv'
TEST_PATH = DATASET_PATH_BASE / 'misogyny_EN' / 'miso_test.tsv'


@DatasetReader.register("misog")
class MisogDatasetReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Optional[Tokenizer] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or \
                {"tokens": SingleIdTokenIndexer()}

        self.random_seed = 0
        np.random.seed(self.random_seed)

    @staticmethod
    def load_misogyny_dataset(path):
        df = pd.read_csv(path, sep='\t')[['text', 'misogynous']]
        npy = df.values
        dataset = np.swapaxes(npy, 0, 1)
        data, labels = dataset
        labels = labels.astype(np.int)
        labels = np.array(labels).reshape(-1, 1)
        return labels, data

    def load_misogyny_train_dataset(self):
        return self.load_misogyny_dataset(TRAIN_PATH)

    def load_misogyny_val_dataset(self):
        return self.load_misogyny_dataset(TEST_PATH)

    def get_data_indices(self, subset):
        np.random.seed(self.random_seed)

        only_train_indices = False
        only_test_indices = False
        if subset == 'train_split':
            subset = 'train'
            only_train_indices = True
        if subset == 'dev_split':
            subset = 'train'
            only_test_indices = True

        if subset == 'train':
            labels, data = self.load_misogyny_train_dataset()
        elif subset == 'test':
            labels, data = self.load_misogyny_val_dataset()
        else:
            raise RuntimeError("{} is not a valid subset".format(subset))

        data_indices = np.array(range(len(data)))
        num_train = math.ceil(TRAIN_VAL_SPLIT_RATIO * len(data_indices))
        print("num_train: ", num_train)
        print("data_indices_len: ", len(data_indices))
        if only_train_indices:
            data_indices = data_indices[:num_train]
        if only_test_indices:
            data_indices = data_indices[num_train:]

        return data_indices, data, labels

    def get_inputs(self, subset, return_labels = False):
        data_indices, data, labels = self.get_data_indices(subset)
        # strings = [None] * len(data_indices)
        # labels = [None] * len(data_indices)
        print(">>> get_inputs: data_indices length is ", len(data_indices))
        print(">>> Iterating")
        print(data[0], labels[0])
        strings = []
        out_labels = []
        for i, idx in enumerate(data_indices):
            sentence = data[idx]
            print(sentence, labels[idx])
            if labels[idx] is None:
                continue
            label = int(labels[idx][0])
            strings.append(sentence)
            out_labels.append(label)

        print(">>> string length: ", len(strings))
        print(">>> labels length: ", len(out_labels))
        # strings = [x for x in strings if x is not None]
        # labels = [x for x in labels if x is not None]
        assert len(strings) == len(out_labels)

        if return_labels:
            print("return labels")
            return strings, out_labels
        return strings

    @overrides
    def _read(self, subset):
        np.random.seed(self.random_seed)
        data_indices, data, labels = self.get_data_indices(subset)
        for idx in data_indices:
            sentence = data[idx]
            if labels[idx] == None:
                continue
            label = int(labels[idx][0])
            if len(sentence) == 0:
                continue
            yield self.text_to_instance(sentence, label)

    def text_to_instance(
            self, string: str, label = None) -> Optional[Instance]:
        tokens = self._tokenizer.tokenize(string)
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields: Dict[str, Field] = {"tokens": text_field}
        if label is not None:
            fields["label"] = LabelField(str(label))
        return Instance(fields)
