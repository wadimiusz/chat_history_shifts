poetry-download:
	curl -sSL https://install.python-poetry.org | python3 -
mypy:
	poetry run python3 -m mypy .
