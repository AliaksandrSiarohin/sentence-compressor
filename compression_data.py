import codecs
import gzip
import io
import json
import sys
from collections import namedtuple

"""Named tuple that defines a token with its fields"""
Token = namedtuple("Token", "form tag stem label")

def load_compression_data(path, limit=None):
    """Load the compression dataset

    The dataset is hosted at:
    http://storage.googleapis.com/sentencecomp/compression-data.json

    Parameters
    -------
    path : str
        The path of the dataset on the local disk

    limit : int
        Limit the number of read sentences

    Returns
    -------
    examples : list of list of Tokens
        A list of examples where each example is a sequence of Token objects
    """

    examples = []
    for compression_json in get_compression_json(path):
        example = compression_json_to_example(compression_json)
        examples.append(example)
        if limit:
            if len(examples) == limit:
                break
    return examples


def get_compression_json(path):
    """Generator that yields a json object for every sentence"""
    with gzip.open(path, "rb") as gzip_in:
        lines = (line.strip() for line in gzip_in)
        data = []
        for line in lines:
            # An empty line signals the beginning of a new example.
            if line == "":
                yield json.loads(" ".join(data))
                data = []
            else:
                data.append(line)
        # Yield the last example if any.
        if data:
            yield json.loads(" ".join(data))


def compression_json_to_example(json_object):
    """Transform the json object in an example which is a list of tokens with
    annotation and labels"""
    sentence = json_object["graph"]["sentence"]
    compression = json_object["compression"]["text"]

    """
    Recover the sentence tokens and their attached annotations
    """
    sentence_tokens = []
    for forms in json_object["graph"]["node"]:
        for word in forms["word"]:
            if word["id"] is not -1:
                sentence_token = (word["id"],
                                  word["form"],
                                  word["tag"],
                                  word["stem"])
                sentence_tokens.append(sentence_token)

    """Sort the tokens by id which correctly reflects the order of the words"""
    sentence_tokens.sort(key=lambda x: x[0])

    """Make sure tokens in the compression can be mapped onto the sentence"""
    compression_tokens = get_compression_tokens(sentence_tokens, compression)

    """Recover the compression labels for the tokens in the sentence"""
    lowercase_tokens = map(lambda x: x[1].lower(), sentence_tokens)
    labels = get_labels(lowercase_tokens, compression_tokens)

    """Produce the example"""
    example = []
    for label, sentence_token in zip(labels, sentence_tokens):
        id, form, tag, stem = sentence_token
        token = Token(form=form, tag=tag, stem=stem, label=label)
        example.append(token)

    return example


def get_labels(sentence_tokens, compression_tokens):
    """Generate the compression labels by comparing the sentence and
    compression tokens

    0 => KEEP the current token
    1 => DELETE the current token
    """
    sentence_tokens_length = len(sentence_tokens)
    compression_tokens_length = len(compression_tokens)
    assert sentence_tokens_length >= compression_tokens_length

    labels = [1] * sentence_tokens_length

    sentence_index = 0
    compression_index = 0
    while sentence_index < len(sentence_tokens) \
            and compression_index < len(compression_tokens):
        sentence_token = sentence_tokens[sentence_index]
        compression_token = compression_tokens[compression_index]
        if sentence_token == compression_token:
            labels[sentence_index] = 0
            sentence_index += 1
            compression_index += 1
        else:
            sentence_index += 1

    return labels


def get_compression_tokens(sentence_tokens, compression):
    """Make sure the compression tokens can be mapped onto the original
    sentence by handling tokens which vary between the two"""

    token_forms = [token[1].lower() for token in sentence_tokens]

    """Collecting the sentence token forms in a set for lookup"""
    token_forms_set = set(token_forms)

    special_abbreviations = {"co": "co.",
                             "corp": "corp.",
                             "ltd": "ltd.",
                             "inc": "inc.",
                             "va": "va.",
                             "wis": "wis.",
                             "pa": "pa.",
                             "st": "st.",
                             "mass": "mass.",
                             "ont": "ont.",
                             "mass": "mass.",
                             "v": "v.",
                             "calif": "calif.",
                             "app": "app."}

    """Normalize compression"""
    compression = compression.lower()

    compression_tokens = []
    for token in compression[:-1].split(" "):
        """Internal tokens do not end with ,"""
        if token.endswith(","):
            token = token[:-1]

        """Handle special abbreviations"""
        if token in special_abbreviations:
            if special_abbreviations[token] in token_forms_set:
                token = special_abbreviations[token]

        if token == "'s":
            compression_tokens.append(token)
        elif token.endswith("'s"):
            compression_tokens.append(token[:-2])
            compression_tokens.append(token[-2:])
        elif token.endswith("n't"):
            compression_tokens.append(token[:-3])
            compression_tokens.append(token[-3:])
        elif token.endswith("%"):
            compression_tokens.append(token[:-1])
            compression_tokens.append(token[-1:])
        elif token == "-":
            pass
        elif token == ":":
            pass
        elif token == ";":
            pass
        elif token == "":
            pass
        elif token == "/":
            pass
        elif token == "!":
            pass
        elif token == "different.":
            compression_tokens.append("different")
        elif token == "prisoner.":
            compression_tokens.append("prisoner")
        elif token == "bar.":
            compression_tokens.append("bar")
        elif token == "declared.":
            compression_tokens.append("declared")
        elif token.startswith("``") and token.endswith("''"):
            compression_tokens.append("``")
            compression_tokens.append(token[2:-2])
            compression_tokens.append("''")
        elif token.startswith("("):
            compression_tokens.append(token[1:])
        elif token.endswith(")"):
            compression_tokens.append(token[:-1])
        elif token.endswith("?"):
            compression_tokens.append(token[:-1])
        else:
            compression_tokens.append(token)

    return compression_tokens
