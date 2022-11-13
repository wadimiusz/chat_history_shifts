import dataclasses
import datetime
import json
import re
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from typing import Any

from gensim.models import Word2Vec
from loguru import logger
from tqdm.auto import tqdm


@dataclasses.dataclass
class Message:
    text: str
    year: int
    user: str

    @classmethod
    def get_message_text(self, message: str | list[Any] | dict[str, Any]) -> str:
        if isinstance(message, str):
            return message
        elif isinstance(message, list):
            return "".join(self.get_message_text(submessage) for submessage in message)
        elif isinstance(message, dict):
            return self.get_message_text(message["text"])

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

            for path in self.args.input:
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

    @staticmethod
    def tokenize(text: str) -> list[str]:
        tokens = [x.lower() for x in re.findall(r"\b\w+\b", text) if x.isalpha()]
        return tokens

    @property
    def models(self) -> list[Word2Vec]:
        if self._models is None:
            models = list()
            for messages in tqdm(self.slices, desc="Training a word2vec model for each slice..."):
                tokenized_sentences = [self.tokenize(sentence) for sentence in messages]
                model = Word2Vec(sentences=tokenized_sentences, vector_size=300, window=5, min_count=1)
                models.append(model)

            self._models = models

        return self._models


def main():
    shifts_detector = ShiftsDetector()


if __name__ == '__main__':
    main()
