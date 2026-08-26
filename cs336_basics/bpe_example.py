
corpus = ("low low low low low \n lower lower widest widest widest \n newest newest newest newest newest newest" )

word_frequency_table : dict[tuple[bytes, ...],int] = {}
combination_frequency_table: dict[tuple[bytes,bytes],int] = {}
vocabulary : dict[int,bytes] = {}
merges : list[tuple[bytes,bytes]] = []

# add pure bytes into vocabulary
for i in range(0,256):
    vocabulary[i] = bytes([i])

# add EOT into vocabulary
vocabulary[256] = "<|endoftext|>".encode("utf-8")

# Converts a string to a tuple of bytes (tuple[bytes, ...]), where each bytes contains only one 'byte'.
def string_to_tuple_of_bytes(string: str) -> tuple[bytes,...]:

    # 把字符串转换为 tuple[bytes] 需要几步？

    # 1. String to Bytes
    a = string
    a_utf8_encoded = a.encode("utf-8")

    # 2. Bytes to list[bytes]
    a_list_of_bytes : list[bytes] = []
    for a_i in a_utf8_encoded:
        a_list_of_bytes.append(bytes([a_i]))

    # 3. list[bytes] to tuple[bytes]
    tuple_of_bytes = tuple(a_list_of_bytes)

    return  tuple_of_bytes

# get pairs in a word
def get_pairs_in_a_word(word: tuple[bytes,...]) -> tuple[tuple[bytes,bytes],...]:
    return tuple(list(zip(word, word[1:])))

def collect_frequency_table():
    for word in corpus.split():
        word_tuple_of_bytes = string_to_tuple_of_bytes(word)
        if ( word_tuple_of_bytes in word_frequency_table):
            word_frequency_table[word_tuple_of_bytes] += 1
        else:
            word_frequency_table[word_tuple_of_bytes] = 1

def get_comb_freq_table_from_word_freq_table():
    for word in word_frequency_table:
        for pair in get_pairs_in_a_word(word):
            if pair in combination_frequency_table:
                combination_frequency_table[pair] += word_frequency_table[word]
            else:
                combination_frequency_table[pair] = word_frequency_table[word]

def get_max_pair() -> tuple[int, tuple[bytes,bytes]]:
    return max(list(zip(combination_frequency_table.values(),combination_frequency_table)))

def merge_word(word : tuple[bytes,...], pair : tuple[bytes, bytes]) -> tuple[bytes,...]:
    new_word_list = []
    i = 0
    while True:
        if i > len(word)-1:
               break
        if word[i] != pair[0]:
            new_word_list.append(word[i])
            i += 1
        else:
            if i > len(word) - 2:
                new_word_list.append(word[i])
                break
            if word[i+1] == pair[1]:
                new_word_list.append(pair[0]+pair[1])
                i += 2
            else:
                    new_word_list.append(word[i])
                    i += 1
    # print("old word:",word,"new word:",new_word_list)
    return tuple(new_word_list)

def on_merge_add_word_to_vocab_and_merges(new_pair : tuple[bytes,bytes]):
    vocabulary[len(vocabulary)] = new_pair[0] + new_pair[1]
    merges.append(new_pair)

def on_merge_update_word_frequency_table():
    old_words : list[tuple[bytes,...]] = []
    new_words : list[tuple[bytes,...]] = []

    for word in word_frequency_table:
        new_word = merge_word(word,merges[-1])
        new_words.append(new_word)
        old_words.append(word)

    for new_word, old_word in zip(new_words,old_words):
        if new_word != old_word:
            word_frequency_table[new_word] = word_frequency_table[old_word]
            del word_frequency_table[old_word]

collect_frequency_table()
# print(word_frequency_table)

for i in range(6):

    get_comb_freq_table_from_word_freq_table()
    # print("word:",word_frequency_table)
    # print("comb:",combination_frequency_table)
    on_merge_add_word_to_vocab_and_merges(get_max_pair()[1])
    on_merge_update_word_frequency_table()
    combination_frequency_table.clear()

print(merges)