import json
import time
import regex as re
import pickle

from cs336_basics import bpe_example


def run_train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) \
        -> tuple[dict[int, bytes], list[tuple[bytes,bytes]]]:
    if "<|endoftext|>" not in special_tokens:
        special_tokens.append("<|endoftext|>")



    # init vocabulary, merges, word_freq_table and comb_freq_table
    word_frequency_table: dict[tuple[bytes, ...], int] = {}
    pair_frequency_table: dict[tuple[bytes, bytes], int] = {}
    vocabulary: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    # add pure bytes into vocabulary
    for i in range(0, 256):
        vocabulary[i] = bytes([i])

    # add EOT into vocabulary
    vocabulary[256] = "<|endoftext|>".encode("utf-8")



    # read corpus from path
    with open(input_path, "r", encoding="utf-8") as f:
        corpus = f.read()

    # segment corpus by special tokens
    chunks = re.split("|".join([re.escape(special_token) for special_token in special_tokens]),corpus)

    # define PAT for finditer
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # collect word_freq_table from chunks:
    for chunk in chunks:
        for word in re.finditer(PAT, chunk):
            word_tuple_of_bytes = bpe_example.string_to_tuple_of_bytes(word[0])
            if (word_tuple_of_bytes in word_frequency_table):
                word_frequency_table[word_tuple_of_bytes] += 1
            else:
                word_frequency_table[word_tuple_of_bytes] = 1

    while len(vocabulary) < vocab_size :

        # update comb_freq_table from word_freq_table
        pair_frequency_table = bpe_example.get_comb_freq_table_from_word_freq_table(word_frequency_table)

        # get new pair
        new_pair = bpe_example.get_max_pair(pair_frequency_table)[1]

        # on merge: update vocab and merges
        vocabulary[len(vocabulary)] = new_pair[0] + new_pair[1]
        merges.append(new_pair)

        # on merge: update word frequency table
        bpe_example.on_merge_update_word_frequency_table(word_frequency_table,new_pair)

        continue

    return vocabulary, merges

if __name__ == "__main__":

    # start_time = time.perf_counter()
    #
    # vocabulary_dict, merge_list = run_train_bpe("/home/mike31nb/Projects/CS336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
    #
    # # TinyStories
    # # serialization in two ways.
    # # 1. pickle, perfect encodings, no damage risk
    # with open("/home/mike31nb/Projects/CS336/assignment1-basics/data/vocabulary.pkl", "wb") as f:  # 注意 "wb"——二进制写
    #     pickle.dump(vocabulary_dict, f)
    # # 2. readable but damage risk
    # with open("/home/mike31nb/Projects/CS336/assignment1-basics/data/vocabulary.json", "w", encoding="utf-8") as f:
    #     json.dump({k: v.decode("utf-8", errors="backslashreplace") for k,v in vocabulary_dict.items()}, f, ensure_ascii=False, indent=4)
    #
    # # merge_list
    # with open("/home/mike31nb/Projects/CS336/assignment1-basics/data/merges.pkl", "wb") as f:  # 注意 "wb"——二进制写
    #     pickle.dump(merge_list, f)
    #
    # with open("/home/mike31nb/Projects/CS336/assignment1-basics/data/merges.json", "w", encoding="utf-8") as f:
    #     json.dump([(pair[0].decode("utf-8", errors="backslashreplace"), pair[1].decode("utf-8", errors="backslashreplace")) for pair in merge_list], f, ensure_ascii=False, indent=4)
    #
    #
    # end_time = time.perf_counter()
    # elapsed_time = end_time - start_time
    # print(f"Time taken: {elapsed_time:.6f} seconds")



    # train with OpenWebText
    start_time = time.perf_counter()

    vocabulary_dict, merge_list = run_train_bpe(
        "/home/mike31nb/Projects/CS336/assignment1-basics/data/owt_train.txt", 32000, ["<|endoftext|>"])

    # serialization in two ways.
    # 1. pickle, perfect encodings, no damage risk
    with open("/home/mike31nb/Projects/CS336/assignment1-basics/data/OpenWebTextvocabulary.pkl", "wb") as f:  # 注意 "wb"——二进制写
        pickle.dump(vocabulary_dict, f)
    # 2. readable but damage risk
    with open("/home/mike31nb/Projects/CS336/assignment1-basics/data/OpenWebTextvocabulary.json", "w", encoding="utf-8") as f:
        json.dump({k: v.decode("utf-8", errors="backslashreplace") for k, v in vocabulary_dict.items()}, f,
                  ensure_ascii=False, indent=4)

    # merge_list
    with open("/home/mike31nb/Projects/CS336/assignment1-basics/data/OpenWebTextmerges.pkl", "wb") as f:  # 注意 "wb"——二进制写
        pickle.dump(merge_list, f)

    with open("/home/mike31nb/Projects/CS336/assignment1-basics/data/OpenWebTextmerges.json", "w", encoding="utf-8") as f:
        json.dump(
            [(pair[0].decode("utf-8", errors="backslashreplace"), pair[1].decode("utf-8", errors="backslashreplace"))
             for pair in merge_list], f, ensure_ascii=False, indent=4)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Time taken for owt: {elapsed_time:.6f} seconds")