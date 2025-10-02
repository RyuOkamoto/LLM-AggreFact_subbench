from scorer import generate_chunks, generate_sliding_chunks, sent_tokenize_with_newlines
from nltk.tokenize import word_tokenize


def test_sent_tokenize_with_newline():
    text_list = [
        "Hello, world! This is the second sentence.",
        "",
        "This is the forth sentence",
        "(here is the part of the forth).",
    ]
    text = "\n".join(text_list)

    expected = [
        "Hello, world!",
        "This is the second sentence.",
        "\n",
        "\n",
        "This is the forth sentence", 
        "\n",
        "(here is the part of the forth).",
    ]

    actual = sent_tokenize_with_newlines(text)
    assert actual == expected


def test_generate_chunks():
    text_list = [
        "This is the 1st sentence.",
        "This is the 2nd sentence.",
        "This is the 3rd sentence.",
        "This is the 4th sentence.",
    ]
    text = " ".join(text_list)
    chunk_size = len(word_tokenize(text_list[0])) + 1

    expected = text_list
    actual = list(generate_chunks(text, chunk_size))
    assert actual == expected


def test_generate_sliding_chunks():
    text_list = [
        "This is the 1st sentence.",
        "This is the 2nd sentence.",
        "This is the 3rd sentence.",
        "This is the 4th sentence.",
    ]
    text = " ".join(text_list)
    chunk_size =  3 * len(word_tokenize(text_list[0])) + 1

    expected = [
        "This is the 1st sentence. This is the 2nd sentence. This is the 3rd sentence.",
        "This is the 2nd sentence. This is the 3rd sentence. This is the 4th sentence.",
    ]
    actual = list(generate_sliding_chunks(text, chunk_size))
    assert actual == expected