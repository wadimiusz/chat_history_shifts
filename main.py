import json
import random
import re
from argparse import ArgumentParser, Namespace
from gensim.models import Word2Vec
from loguru import logger
from typing import Any


class ShiftsDetector:
    def __init__(self):
        self._args: Namespace | None = None
        self._messages: Namespace | None = None
        self._model: Word2Vec | None = None

    @property
    def args(self) -> Namespace:
        if self._args is None:
            parser = ArgumentParser()
            parser.add_argument("-i", "--input", required=True)
            args = parser.parse_args()
            self._args = args

        return self._args

    @property
    def messages(self) -> list[str]:
        if self._messages is None:
            logger.info("Loading messages from the .json file...")
            with open(self.args.input) as f:
                history = json.load(f)

            messages = list()
            for message in history["messages"]:
                message_text = self.get_message_text(message)
                if len(message_text) > 0:
                    messages.append(message_text)

            self._messages = messages
            logger.info("Messages loaded")

        return self._messages

    @staticmethod
    def tokenize(text: str) -> list[str]:
        tokens = [x.lower() for x in re.findall(r"\b\w+\b", text) if x.isalpha()]
        return tokens

    def get_message_text(self, message: str | list[Any] | dict[Any, Any]) -> str:
        if isinstance(message, str):
            return message
        elif isinstance(message, list):
            return "".join(self.get_message_text(submessage) for submessage in message)
        elif isinstance(message, dict):
            return self.get_message_text(message["text"])

        raise TypeError(f"The `message` argument is of type {type(message)}. Expected string, list, or dict")

    @property
    def model(self) -> Word2Vec:
        if self._model is None:
            logger.info("Tokenizing sentences...")
            tokenized_sentences = [self.tokenize(sentence) for sentence in self.messages]
            logger.info(f"Training a word2vec model on {len(tokenized_sentences)} messages...")
            model = Word2Vec(sentences=tokenized_sentences, vector_size=300, window=5, min_count=1)
            logger.info("Training finished.")
            self._model = model

        return self._model


def main():
    shifts_detector = ShiftsDetector()


if __name__ == '__main__':
    main()
