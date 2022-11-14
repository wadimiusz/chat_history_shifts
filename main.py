import dataclasses
import datetime
import json
import re
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from functools import lru_cache
from typing import Any

import numpy as np
import scipy.linalg
from gensim.models import Word2Vec
from loguru import logger
from pymorphy2 import MorphAnalyzer
from tqdm.auto import tqdm


@lru_cache()
def lemmatize(morph_analyzer: MorphAnalyzer, word: str) -> str:
    return morph_analyzer.parse(word)[0].normal_form


@dataclasses.dataclass
class Message:
    text: str
    year: int
    user: str

    @classmethod
    def get_message_text(cls, message: str | list[Any] | dict[str, Any]) -> str:
        if isinstance(message, str):
            return message
        elif isinstance(message, list):
            return "".join(cls.get_message_text(submessage) for submessage in message)
        elif isinstance(message, dict):
            return cls.get_message_text(message["text"])

        raise TypeError(f"The `message` argument is of type {type(message)}. Expected string, list, or dict")

    @staticmethod
    def get_message_year(message_dict: dict[str, Any]) -> int:
        time = datetime.datetime.fromisoformat(message_dict["date"])
        return time.year

    @staticmethod
    def get_message_user(message_dict: dict[str, Any]) -> str:
        return message_dict["from"]

    @classmethod
    def from_dict(cls, message_dict: dict[str, Any]) -> "Message":
        text = cls.get_message_text(message_dict)
        user = cls.get_message_user(message_dict)
        year = cls.get_message_year(message_dict)

        return cls(text=text, user=user, year=year)


class ShiftsDetector:
    def __init__(self):
        self._args: Namespace | None = None
        self._name_to_slice: dict[str, list[str]] | None = None
        self._name_to_model: dict[str, Word2Vec] | None = None
        self.morph_analyzer = MorphAnalyzer()

    @property
    def args(self) -> Namespace:
        if self._args is None:
            parser = ArgumentParser()
            parser.add_argument("-i", "--inputs", required=True, nargs="+")
            parser.add_argument("-u", "--users", default=None, nargs="+")
            args = parser.parse_args()
            self._args = args

        return self._args

    @property
    def name_to_slice(self) -> dict[str, list[str]]:
        if self._name_to_slice is None:
            name_to_slice: dict[str, list[str]] = defaultdict(list)
            logger.info("Loading messages from the JSON files...")
            all_messages: list[list[dict[str, Any]]] = list()

            for path in self.args.inputs:
                with open(path) as f:
                    history = json.load(f)

                for chat in history["chats"]["list"]:
                    all_messages.append(chat["messages"])

            for messages in all_messages:
                for message_dict in messages:
                    if message_dict["type"] == "message":
                        message = Message.from_dict(message_dict)
                        if len(message.text) > 0 and (self.args.users is None or message.user in self.args.users):
                            name_to_slice[f"{message.year}"].append(message.text)

            self._name_to_slice = {key: name_to_slice[key] for key in sorted(name_to_slice.keys(), key=lambda x: int(x))}
            logger.info("Messages loaded")

        return self._name_to_slice

    def normalize(self, text: str) -> list[str]:
        tokens = [lemmatize(morph_analyzer=self.morph_analyzer, word=token).lower() for token in re.findall(r"\b\w+\b", text) if token.isalpha()]
        return tokens

    @property
    def name_to_model(self) -> dict[str, Word2Vec]:
        if self._name_to_model is None:
            name_to_model = dict()
            for name, messages in tqdm(self.name_to_slice.items(), desc="Training a word2vec model for each slice..."):
                logger.info(f"Now preparing a model for {name}")
                tokenized_sentences = [self.normalize(sentence) for sentence in messages]
                texts = list()
                for i in range(len(tokenized_sentences)):
                    new_text = list()
                    for sentence in tokenized_sentences[i-20:i+20]:
                        new_text.extend(sentence)
                    texts.append(new_text)
                model = Word2Vec(sentences=tqdm(texts), vector_size=300, window=20, min_count=10)
                name_to_model[name] = model

            self._name_to_model = name_to_model

        return self._name_to_model

    def get_changes_between_two_models(self, name1: str, name2: str, top_n: int) -> list[str]:
        keyed_vectors_1 = self.name_to_model[name1].wv
        keyed_vectors_2 = self.name_to_model[name2].wv

        keys_1 = set(keyed_vectors_1.key_to_index.keys())
        keys_2 = set(keyed_vectors_2.key_to_index.keys())

        common_keys = sorted(keys_1 & keys_2)

        matrix_1 = np.vstack([keyed_vectors_1.get_vector(key) for key in common_keys])
        matrix_2 = np.vstack([keyed_vectors_2.get_vector(key) for key in common_keys])

        mapping, _ = scipy.linalg.orthogonal_procrustes(matrix_1, matrix_2)
        matrix_1_aligned = matrix_1 @ mapping

        scores = (matrix_1_aligned * matrix_2).sum(-1)
        indices_sorted_by_score = scores.argsort()

        changed_words_indices = indices_sorted_by_score[:top_n]

        changed_words = [common_keys[idx] for idx in changed_words_indices]
        return changed_words


def main():
    shifts_detector = ShiftsDetector()
    names = list(shifts_detector.name_to_slice.keys())
    for i in range(len(names) - 1):
        name1, name2 = names[i], names[i+1]
        words = shifts_detector.get_changes_between_two_models(name1, name2, 10)
        logger.info(f"The most changed words from {name1} to {name2} are: {', '.join(words)}")


if __name__ == '__main__':
    main()
