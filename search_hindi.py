import string
import time
import sys
import os
import re
import pickle
import Stemmer
from collections import OrderedDict
from ordered_set import OrderedSet
from math import log

start_time = time.time()
temp = time.time()
stop_words_list = []
with open('./hindi_stopwords.pkl', 'rb') as _f:
    stop_words_list = pickle.load(_f)
curr_folder_path = '../../results_hindi/'
if len(sys.argv) != 2:
    print("Invalid arguments passed :(")
    sys.exit()
path_to_query_strings_file = sys.argv[1]
tag_set = set("tibclr")
if not os.path.isfile(path_to_query_strings_file):
    print("Invalid input query file")
    sys.exit()
with open(curr_folder_path + 'inverted_index/secondary_index.pkl', 'rb') as _f:
    secondary_index = pickle.load(_f)
if os.path.isfile("queries_op.txt"):
    os.remove("queries_op.txt")
ss = Stemmer.Stemmer('hindi')
to_be_discarded = [".", '-', "'", "infobox", "ref", "amp", "quot", "apos", "url", "cite", "name", "title", "website", "file", "jpg", "png", "jpeg", "category", "references", "reflist", "navboxes", "सन्दर्भ", "श्रेणी", "टिप्पणीसूची"]
stop_words = set(stop_words_list + list(string.punctuation))
INDEX_PARTS_CACHE_LIMIT, TITLES_CACHE_LIMIT, MINN, MAXX, ACCESSED_TOKENS_CACHE_LIMIT, N, FREQUENCY_THRESHOLD, QUERY_TOKENS_BOUND = 50, 6, 1, len(secondary_index), 120, 284425, 155000000, 40
pseudo_cache_index_parts, pseudo_cache_titles, pseudo_cache_accessed_tokens, best_results = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
RELATIVE_WEIGHTS = {"t": 30, "i": 10, "b": 0.9, "c": 6, "l": 4.5, "r": 1.5}
# RELATIVE_WEIGHTS = {"t": 3, "i": 1.4, "b": 0.6, "c": 1.1, "l": 0.9, "r": 0.7}
encodings_2 = ['-', 'a', 'e', 'i', 't', 'o', 'r', 'n', 's', 'l', '1', '0', 'c', '|', 'd', 'm', '2', 'u', '.', 'h', '=', 'w', 'p', 
               '/', 'q', 'g', 'b', '3', '9', '4', 'f', '5', '8', '7', '6', 'k', 'v', 'y', 'j', '_', 'x', 'z', ':', ',', '+', "'", 
               '~', '\\', '^', ' ', '!', '#', '$', '%', '&', '(', ')', '*', ';', '<', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 
               'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '`', '{', '}']
encodings = [c for c in encodings_2 if not c in ("t", "i", "b", "c", "l", "r", "|", "$")]
int_decoder = {encodings[d]: d for d in range(len(encodings))}
inverted_index = {}

def decode_int(given_str):
    num, val = 0, 1
    for j in range(len(given_str)):
        num = num + (val*int_decoder[given_str[j]])
        val *= 86
    return num

def get_fieldwise_split(query):
    curr_field, curr_string, i, field_split = "d", "", 0, {} 
    while i < len(query):
        if i < len(query) - 1 and query[i] in tag_set and query[i+1] == ':':
            field_split[curr_field] = curr_string
            curr_string, curr_field = "", query[i]
            i += 1
        else:
            curr_string += query[i]
        i += 1
    if len(curr_string) > 0:
        field_split[curr_field] = curr_string
    return field_split

def process_tokens_list(query_type, tokens_list):
    global pseudo_cache_accessed_tokens, best_results
    for token in tokens_list:
        if pseudo_cache_accessed_tokens.get(token) is None:
            continue
        postings_list_string = pseudo_cache_accessed_tokens[token]
        segment_wise_postings = postings_list_string.split('$')
        if query_type != 'd':
            doc_frequency = int(postings_list_string.count(query_type))
        else:
            doc_frequency = sum([int(_s.count('|') + 1) for _s in segment_wise_postings])
        if doc_frequency == 0:
            continue
        docs_done = 0
        idf_score = log(N / doc_frequency, 1.3)
        for segment in segment_wise_postings:
            if docs_done >= FREQUENCY_THRESHOLD:
                break
            curr_doc_id, decoded_list = 0, segment.split('|')
            for d in decoded_list:
                curr_field, curr_string, i, tf_score = "d", "", 0, 0
                while i < len(d):
                    if i < len(d) - 1 and d[i] in tag_set:
                        if curr_field == "d":
                            curr_doc_id += decode_int(curr_string)
                        elif (query_type == 'd') or (query_type != 'd' and curr_field == query_type):
                            tf_score += (decode_int(curr_string) * RELATIVE_WEIGHTS[curr_field])
                        curr_string, curr_field = "", d[i]
                    else:
                        curr_string += d[i]
                    i += 1
                if (len(curr_string) > 0) and ((query_type == 'd') or (query_type != 'd' and curr_field == query_type)):
                    tf_score += (decode_int(curr_string) * RELATIVE_WEIGHTS[curr_field])
                if tf_score > 0:
                    total_score = (1 + log(tf_score, 2.7)) * idf_score
                    if best_results.get(curr_doc_id) is None:
                        best_results[curr_doc_id] = total_score
                    else:
                        best_results[curr_doc_id] += total_score
                docs_done += 1
                if docs_done >= FREQUENCY_THRESHOLD:
                    break

def pre_process_query(given_string):
    if len(given_string) == 0:
        return []
    words = re.split("\||/|\=|_|\:|\+|,|~|\^|#|\[|\]|\(|\)|\{|\}|<|>|\!|&|;|\?|\*|%|\$|@|`|\s+", given_string.lower())
    stripped_tokens = [tok.strip("|/=.-_:'+,~^#[](){}<>!&;?*%$@`") if tok.strip("|/=.-_:'+,~^#[](){}<>!&;?*%$@`") not in stop_words else tok for tok in words]
    return [ss.stemWord(t) for t in stripped_tokens if (len(t) > 1) and (len(t) <= 40) and (not t.isalpha()) and (t not in to_be_discarded)]

query_op = open("queries_op.txt", "a")
with open(path_to_query_strings_file, 'r') as q_f:
    unique_tokens, answers_dict = set(), OrderedDict()
    # print("Time taken for initial loading: ", time.time() - temp)
    # temp = time.time()
    for query_string in q_f.readlines():
        # print(query_string)
        fieldwise_split = get_fieldwise_split(query_string)
        pre_processed_query = {k: pre_process_query(fieldwise_split[k]) for k in fieldwise_split.keys()}
        for _v in pre_processed_query.values():
            for _val in _v:
                unique_tokens.add(_val)
        sorted_unique_tokens = OrderedSet(sorted(unique_tokens))
        # print("Time taken for pre-process and sorting toks: ", time.time() - temp)
        # temp = time.time()
        curr_tok_count = 0
        for _t in sorted_unique_tokens:
            curr_tok_count += 1
            if pseudo_cache_accessed_tokens.get(_t) is not None or curr_tok_count >= QUERY_TOKENS_BOUND:
                continue
            minn, maxx, mid = MINN, MAXX, 0
            while maxx > minn:
                mid = minn + ((maxx - minn) // 2)
                if _t > secondary_index[mid]:
                    minn = mid + 1
                else:
                    maxx = mid
            if pseudo_cache_index_parts.get(minn) is None:
                with open(curr_folder_path + f'inverted_index/index_version_{minn}.pkl', 'rb') as p_f:
                    pseudo_cache_index_parts[minn] = pickle.load(p_f)
                if len(pseudo_cache_index_parts) > INDEX_PARTS_CACHE_LIMIT:
                    del pseudo_cache_index_parts[next(iter(pseudo_cache_index_parts.keys()))]
            if pseudo_cache_index_parts[minn].get(_t) is not None:
                pseudo_cache_accessed_tokens[_t] = pseudo_cache_index_parts[minn][_t]
                if len(pseudo_cache_accessed_tokens) > ACCESSED_TOKENS_CACHE_LIMIT:
                    del pseudo_cache_accessed_tokens[next(iter(pseudo_cache_accessed_tokens.keys()))]
        # print("Time taken for loading and cache update: ", time.time() - temp)
        # temp = time.time()             
        for _k, _v in pre_processed_query.items():
            process_tokens_list(_k, _v)
        best_results = OrderedDict(sorted(best_results.items(), key=lambda s: -1*s[1]))
        while len(best_results) > 10:
            best_results.popitem()
        top_docs = sorted(best_results.keys())
        # print("Time taken for obtaining top 10: ", time.time() - temp)
        # temp = time.time()
        for _d in top_docs:
            title_dict_num = ((_d - 1) // 100000) + 1
            offset = (_d - 1) % 100000
            if pseudo_cache_titles.get(title_dict_num) is None:
                with open(curr_folder_path + f'titles/titles_part_{title_dict_num}.pkl', 'rb') as _t:
                    pseudo_cache_titles[title_dict_num] = pickle.load(_t)
                if len(pseudo_cache_titles) > TITLES_CACHE_LIMIT:
                    del pseudo_cache_titles[next(iter(pseudo_cache_titles.keys()))]
            answers_dict[_d] = pseudo_cache_titles[title_dict_num][offset]
        for _k in best_results.keys():
            query_op.write(str(_k) + ", " + answers_dict[_k] + '\n')                       
        unique_tokens.clear()
        best_results.clear()
        answers_dict.clear()
        # print("Time taken for writing to file and clearing: ", time.time() - temp)
        # temp = time.time()
        end_time = time.time()
        query_op.write(str(end_time - start_time) + "\n\n")
        start_time = time.time()
        # print()
query_op.close()