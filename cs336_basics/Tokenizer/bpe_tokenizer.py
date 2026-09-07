import regex as re
import pickle
from collections.abc import Iterable, Iterator
from cs336_basics.Tokenizer import bpe_example


class tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.special_tokens = [] if special_tokens is None else special_tokens
        self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        self.vocab = vocab
        self.merges = merges
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.cache = {}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        with open(vocab_filepath, "rb") as f:
            vocabulary = pickle.load(f)
        return cls(vocabulary,merges,special_tokens)

    def encode(self, text: str) -> list[int]:

        # segment corpus by special tokens
        if len(self.special_tokens) != 0:
            chunks = re.split("(" + "|".join([re.escape(special_token) for special_token in self.special_tokens]) + ")", text)
        else:
            chunks = [text]

        # define PAT for finditer
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        new_words = []
        # collect word_freq_table from chunks:
        for chunk in chunks:
            if chunk in self.special_tokens:
                new_words.append((chunk.encode("utf-8"),))
            else:
                # word is string
                for word in re.finditer(PAT, chunk):
                    # keep string word
                    word_init = word[0]
                    # check if cached
                    if word_init in self.cache:
                        new_words.append(self.cache[word_init])
                    else:
                        # change word into tuple of bytes
                        word = bpe_example.string_to_tuple_of_bytes(word[0])
                        # word is merged tuple of bytes
                        for pair in self.merges:
                            word = bpe_example.merge_word(word, pair)
                        # add tuple of bytes as word into words
                        new_words.append(word)
                        # update cache
                        self.cache[word_init] = word

        # word_to_index
        indices = [self.inverse_vocab[token] for new_word in new_words for token in new_word]
        return indices

    def decode(self, ids: list[int]) -> str:
        # get list of bytes
        words = [self.vocab[index] for index in ids]
        return b"".join(words).decode("utf-8",errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for little_corpus in iterable:
            indices = self.encode(little_corpus)
            for index in indices:
                yield index

    def encode_iterable_return_in_list(self, iterable: Iterable[str]) -> list[int]:
        for little_corpus in iterable:
            indices = self.encode(little_corpus)
            yield list(indices)