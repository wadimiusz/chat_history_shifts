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
        self._slices: list[list[str]] | None = None
        self._models: list[Word2Vec] | None = None
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
    def slices(self) -> list[list[str]]:
        if self._slices is None:
            year_to_messages: dict[int, list[str]] = defaultdict(list)
            logger.info("Loading messages from the JSON files...")
            all_messages: list[list[dict[str, Any]]] = list()

            for path in self.args.inputs:
                with open(path) as f:
                    history = json.load(f)

                all_messages.append(history["messages"])

            for messages in all_messages:
                for message_dict in messages:
                    if message_dict["type"] == "message":
                        message = Message.from_dict(message_dict)
                        if len(message.text) > 0 and (self.args.users is None or message.user in self.args.users):
                            year_to_messages[message.year].append(message.text)

            self._slices = [year_to_messages[year] for year in sorted(year_to_messages.keys())]
            logger.info("Messages loaded")

        return self._slices

    def normalize(self, text: str) -> list[str]:
        tokens = [lemmatize(morph_analyzer=self.morph_analyzer, word=token).lower() for token in re.findall(r"\b\w+\b", text) if token.isalpha()]
        return tokens

    @property
    def models(self) -> list[Word2Vec]:
        if self._models is None:
            models = list()
            for messages in tqdm(self.slices, desc="Training a word2vec model for each slice..."):
                tokenized_sentences = [self.normalize(sentence) for sentence in messages]
                model = Word2Vec(sentences=tokenized_sentences, vector_size=300, window=5, min_count=10)
                models.append(model)

            self._models = models

        return self._models

    def get_changes_between_two_models(self, model_i: int, model_j: int, top_n: int) -> list[str]:
        keyed_vectors_i = self.models[model_i].wv
        keyed_vectors_j = self.models[model_j].wv

        keys_i = set(keyed_vectors_i.key_to_index.keys())
        keys_j = set(keyed_vectors_j.key_to_index.keys())

        common_keys = sorted(keys_i & keys_j)

        matrix_i = np.vstack([keyed_vectors_i.get_vector(key) for key in common_keys])
        matrix_j = np.vstack([keyed_vectors_j.get_vector(key) for key in common_keys])

        mapping, _ = scipy.linalg.orthogonal_procrustes(matrix_i, matrix_j)
        matrix_i_aligned = matrix_i @ mapping

        scores = (matrix_i_aligned * matrix_j).sum(-1)
        indices_sorted_by_score = scores.argsort()

        changed_words_indices = indices_sorted_by_score[:top_n]

        changed_words = [common_keys[idx] for idx in changed_words_indices]
        return changed_words


def main():
    shifts_detector = ShiftsDetector()
    words = shifts_detector.get_changes_between_two_models(-1, -2, 10)
    logger.info(f"The most changed words are: {', '.join(words)}")


if __name__ == '__main__':
    main()
