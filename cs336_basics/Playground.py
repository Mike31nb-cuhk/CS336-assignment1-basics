import pickle
import time
import cs336_basics.bpe_tokenizer
import numpy as np

with open("/home/mike31nb/Projects/CS336/assignment1-basics/data/merges.pkl", "rb") as f:
    merges = pickle.load(f)
with open("/home/mike31nb/Projects/CS336/assignment1-basics/data/vocabulary.pkl", "rb") as f:
    vocabulary = pickle.load(f)
with open("/home/mike31nb/Projects/CS336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt", "r") as f:
    tiny_story = f.read()

def see_tiny_story():
    with open("/home/mike31nb/Projects/CS336/assignment1-basics/data/OpenWebTextIndicies.pkl", "rb") as f:
        return pickle.load(f)

def encode_tiny_story():
    articles = tiny_story.split("<|endoftext|>")
    output = []
    tokenizer = cs336_basics.bpe_tokenizer.tokenizer(vocabulary, merges, ["<|endoftext|>"])

    for index in tokenizer.encode_iterable(articles):
        output.append(index)

    np_array = np.array(output,dtype=np.uint16)
    with open("/home/mike31nb/Projects/CS336/assignment1-basics/data/OpenWebTextIndicies.pkl", "wb") as f:  # 注意 "wb"——二进制写
        pickle.dump(np_array, f)
def tokenizer_performance_analysis_iter():

    start_time = time.perf_counter()

    articles = tiny_story.split("<|endoftext|>")
    output = []
    tokenizer = cs336_basics.bpe_tokenizer.tokenizer(vocabulary, merges, ["<|endoftext|>"])

    for index in tokenizer.encode_iterable(articles):
        output.append(index)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # throughput: 1171647.231850285 bytes per second when encoding TinyStories

    num_of_bytes = 0
    num_of_token = 0

    for i in articles:
        num_of_bytes += len(i.encode("utf-8"))

    num_of_token += len(output)

    print(num_of_bytes / num_of_token)
    print(num_of_bytes / elapsed_time)


def tokenizer_performance_analysis_iter_return_list():

    start_time = time.perf_counter()

    articles = tiny_story.split("<|endoftext|>")
    output = []
    tokenizer = cs336_basics.bpe_tokenizer.tokenizer(vocabulary, merges, ["<|endoftext|>"])

    for indices_list in tokenizer.encode_iterable_return_in_list(articles):
        output.append(indices_list)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # throughput: 1177163.1207174752 bytes per second when encoding TinyStories

    num_of_bytes = 0
    num_of_token = 0

    for i,j in zip(articles,output):
        num_of_bytes += len(i.encode("utf-8"))
        num_of_token += len(j)

    print(num_of_bytes / num_of_token)
    print(num_of_bytes / elapsed_time)

def tokenizer_performance_analysis_naive():

    start_time = time.perf_counter()

    articles = tiny_story.split("<|endoftext|>")
    tokenizer = cs336_basics.bpe_tokenizer.tokenizer(vocabulary,merges,["<|endoftext|>"])
    total_text = "".join(articles)
    total_code = tokenizer.encode(total_text)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # owt_valid.txt with TinyStories-trained bpe_tokenizer
    # byte number: 43634
    # token number: 10733
    # compression rate: 4.065405757942793

    # owt_valid.txt with TinyStories-trained bpe_tokenizer
    # byte number: 235056
    # token number: 71654
    # compression rate: 3.280430959890585

    # throughput: 1182463.9323805906 bytes per second when encoding TinyStories
    # time taken to tokenize Pile dataset(): 749145.0525823586 second = 8.67 days
    # comment: obviously too slow but yeah

    print(len(total_text.encode("utf-8")))
    print(len(total_code))
    print(len(total_text.encode("utf-8"))/len(total_code))
    print(len(total_text.encode("utf-8")) /elapsed_time)
    print(825*1024*1024*1024 / (len(total_text.encode("utf-8")) /elapsed_time))

if __name__ == "__main__":
    OpenWebTextIndicies = see_tiny_story()