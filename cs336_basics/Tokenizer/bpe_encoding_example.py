import regex as re
import pickle
from cs336_basics.Tokenizer import bpe_example

special_tokens = ["<|endoftext|>"]

with open("/data/merges.pkl", "rb") as f:
    merges = pickle.load(f)
with open("/data/vocabulary.pkl", "rb") as f:
    vocabulary = pickle.load(f)

inverse_vocab = {v : k for k,v in vocabulary.items()}
toy_corpus = "你好<|endoftext|>"

def encoding(corpus : str) -> list[int]:

    # segment corpus by special tokens
    chunks = re.split( "(" + "|".join( [re.escape(special_token) for special_token in special_tokens]) + ")",corpus)

    # define PAT for finditer
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    new_words = []
    # collect word_freq_table from chunks:
    for chunk in chunks:
        if chunk  in special_tokens:
            new_words.append((chunk.encode("utf-8"),))
        else:
            # word is string
            for word in re.finditer(PAT, chunk):
                # change word into tuple of bytes
                word = bpe_example.string_to_tuple_of_bytes(word[0])
                # word is merged tuple of bytes
                for pair in merges:
                    word = bpe_example.merge_word(word, pair)
                # add tuple of bytes as word into words
                new_words.append(word)
    # word_to_index
    indices = [inverse_vocab[token] for new_word in new_words for token in new_word]
    return indices

def decoding(indices: list[int]) -> str:
    # get list of bytes
    words = [vocabulary[index] for index in indices]
    return b"".join(words).decode("utf-8")

if __name__ == "__main__":
    print(encoding(toy_corpus))
    print(decoding(encoding(toy_corpus)))