# LIMA-FR

This project is dedicated to translating the LIMA (Less Is More for Alignment) dataset from English to French using OpenAI's API (`gpt-3.5-turbo`).

## Source Dataset

LIMA (Less Is More for Alignment) can be found [here](https://arxiv.org/pdf/2305.11206.pdf).

## Setup

1.  Ensure you have all required libraries installed:

```bash
pip install openai datasets
```

2.  Set up your OpenAI API key to access the model for translations.

## Usage

To translate the LIMA dataset:

```bash
python translate_lima.py
```

This script will:

- Load the LIMA dataset (train and test splits).
- Translate each item from English to French using the OpenAI API.
- Save translated items in `lima-fr_train.jsonl` and `lima-fr_test.jsonl`.
- Save any missed translations or errors in `missed_entries_train.jsonl` and `missed_entries_test.jsonl`.

## Licensing

The LIMA dataset's licensing rules are applied here. If the LIMA source data uses a stricter license than CC BY-NC-SA, then this project follows the same. Otherwise, it abides by the CC BY-NC-SA license.

## Contribution

Suggestions and contributions are always welcome. Please ensure that any changes made do not conflict with the original licensing of the LIMA dataset.
