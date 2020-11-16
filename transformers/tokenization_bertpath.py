import collections
import logging
import os
from typing import List, Optional
# import unicodedata

# import tokenizers as tk

from .tokenization_utils import PreTrainedTokenizer, PreTrainedTokenizerFast


logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "bertpath-mlmnspsp-ace": "conf/ace/bertpath-mlmnspsp-ace-vocab.txt",
        "bertpath-mlmnspsp-wiki": "conf/wikipedia/bertpath-mlmnspsp-wiki-vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "bertpath-mlmnspsp-ace": 512,
    "bertpath-mlmnspsp-wiki": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "bertpath-mlmnspsp-ace": {'do_lower_case': True},  # can not work! still do lower case, did not find why????
    "bertpath-mlmnspsp-wiki": {'do_lower_case': True},  # can not work! still do lower case, did not find why????
}


def load_vocab(vocab_file, do_lower_case=False):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        if do_lower_case:
            token = token.lower()
        vocab[token] = index
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    # print('text', text)
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BERTPathTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
            self,
            vocab_file,
            do_lower_case=False,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            **kwargs
    ):
        super(BERTPathTokenizer, self).__init__(
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )
        # self.max_len_single_sentence = self.max_len - 2  # take into account special tokens
        # self.max_len_sentences_pair = self.max_len - 3  # take into account special tokens
        self.do_lower_case = do_lower_case

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BERTPathTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file)
            )
        self.vocab = load_vocab(vocab_file, do_lower_case)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        print(self.vocab)

        # print('do_lower_case lml ', self.init_kwargs.get("do_lower_case", False))

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        if self.do_lower_case:
            text = text.lower()
        return whitespace_tokenize(text)

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        # print(token)
        # print(self.vocab.get(token, self.vocab.get(self.unk_token)))
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A BERT sequence has the following format:

        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True if the token list is already formatted with special tokens for the model

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 0 for a special token, 1 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        A BERT sequence pair mask has the following format:

        ::

            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |

        if token_ids_1 is None, only returns the first portion of the mask (0's).

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def create_element_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        element_ids = list()

        # seq 1
        element_ids.append(self.pad_token_id) # cls
        for idx in range(len(token_ids_0)):
            if idx % 2 == 0:
                # node
                element_ids.append(1)
            else:
                # edge
                element_ids.append(2)
        element_ids.append(self.pad_token_id)  # sep

        # seq 2
        if token_ids_1 is not None:
            for idx in range(len(token_ids_1)):
                if idx % 2 == 0:
                    # node
                    element_ids.append(1)
                else:
                    # edge
                    element_ids.append(2)
            element_ids.append(self.pad_token_id)  # sep
        return element_ids

    def save_vocabulary(self, vocab_path):
        """
        Save the sentencepiece vocabulary (copy original file) and special tokens file to a directory.

        Args:
            vocab_path (:obj:`str`):
                The directory in which to save the vocabulary.

        Returns:
            :obj:`Tuple(str)`: Paths to the files saved.
        """
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, VOCAB_FILES_NAMES["vocab_file"])
        else:
            vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)


# class PathTokenizerFast(PreTrainedTokenizerFast):
#     vocab_files_names = VOCAB_FILES_NAMES
#     pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
#     pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
#     max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
#
#     def __init__(
#             self,
#             vocab_file,
#             do_lower_case=False,
#             unk_token="[UNK]",
#             sep_token="[SEP]",
#             pad_token="[PAD]",
#             cls_token="[CLS]",
#             mask_token="[MASK]",
#             max_length=None,
#             pad_to_max_length=False,
#             stride=0,
#             truncation_strategy="longest_first",
#             add_special_tokens=True,
#             **kwargs
#     ):
#         super(PathTokenizer, self).__init__(
#             unk_token=unk_token,
#             sep_token=sep_token,
#             pad_token=pad_token,
#             cls_token=cls_token,
#             mask_token=mask_token,
#             **kwargs,
#         )
#
#         self._tokenizer = tk.Tokenizer(tk.models.WhiteSpace.from_files(vocab_file, unk_token=unk_token))

