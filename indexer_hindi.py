import xml.etree.ElementTree as ET
from bz2file import BZ2File
import re
import time
import os
import sys
import pickle
import string
import shutil
import Stemmer
from queue import PriorityQueue
from collections import Counter, OrderedDict

stop_words_list = []
with open('./hindi_stopwords.pkl', 'rb') as _f:
    stop_words_list = pickle.load(_f)
ss = Stemmer.Stemmer('hindi')
previous_postings, inverted_index, encoded_ints, chunk_storage, titles_list, secondary_index = {}, OrderedDict(), {}, {}, [], OrderedDict()
ARTICLE_LIMIT, ENCODED_INTS_LIMIT, CHUNK_LIMIT = 66000, 900000, 1
to_be_discarded = [".", '-', "'", "infobox", "ref", "amp", "quot", "apos", "url", "cite", "name", "title", "website", "file", "jpg", "png", "jpeg", "category", "references", "reflist", "navboxes", "सन्दर्भ", "श्रेणी", "टिप्पणीसूची"]
stop_words = set(stop_words_list + list(string.punctuation))
current_title_version, final_index_version, intermediate_index_version = 1, 1, 1
intermediate_index_length = 0
TITLES_PER_FILE, INDEX_LENGTH_LIMIT, BUFFER_LIMIT = 100000, 4010000, 1610000
merge_sort_dict = OrderedDict()

encodings_2 = ['-', 'a', 'e', 'i', 't', 'o', 'r', 'n', 's', 'l', '1', '0', 'c', '|', 'd', 'm', '2', 'u', '.', 'h', '=', 'w', 'p', 
               '/', 'q', 'g', 'b', '3', '9', '4', 'f', '5', '8', '7', '6', 'k', 'v', 'y', 'j', '_', 'x', 'z', ':', ',', '+', "'", 
               '~', '\\', '^', ' ', '!', '#', '$', '%', '&', '(', ')', '*', ';', '<', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 
               'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '`', '{', '}']
encodings = [c for c in encodings_2 if not c in ("t", "i", "b", "c", "l", "r", "|", "$")]
int_encoder = {d: encodings[d] for d in range(len(encodings))}

if os.path.isdir('../../results_hindi/intermediates'):
    shutil.rmtree('../../results_hindi/intermediates')
os.mkdir('../../results_hindi/intermediates')
if os.path.isdir('../../results_hindi/inverted_index'):
    shutil.rmtree('../../results_hindi/inverted_index')
os.mkdir('../../results_hindi/inverted_index')
if os.path.isdir('../../results_hindi/titles'):
    shutil.rmtree('../../results_hindi/titles')
os.mkdir('../../results_hindi/titles')

def encode_int(given_num):
    num, req = given_num, ""
    while num > 0:
        req += int_encoder[num % 86]
        num = num // 86
    return req

def get_split_string(doc_id, title, given_text):
    global chunk_storage
    text = given_text
    l, c, r, b, b2, i, r2 = '', '', '', '', '', '', ''
    info_match = re.search("\{\{\s*Infobox", text, re.IGNORECASE)
    if info_match != None:
        info_start, info_idx = info_match.start(), info_match.end()
        final_idx, ctr = info_idx, 2
        for i in range(info_idx, len(text)):
            if text[i] == '{':
                ctr += 1
            elif text[i] == '}':
                ctr -= 1
            if ctr == 0:
                final_idx = i
                break
        i, b = text[info_start:final_idx+1], text[:info_start]
        text = text[final_idx + 1:]
    categories_match = re.search("\[\[\s*श्रेणी\s*:", text)
    if categories_match != None:
        c_idx = categories_match.start()
        c = text[c_idx:]
        text = text[:c_idx]
    links_match = re.search("==\s*बाहरी\s*कड़ियाँ\s*==", text)
    if links_match != None:
        external_links_idx = links_match.start()
        l = text[external_links_idx:]
        text = text[:external_links_idx]
    ref_match = re.search("==\s*सन्दर्भ\s*==", text)
    if ref_match != None:
        ref_start = ref_match.start()
        r = text[ref_start:]
        text = text[:ref_start] 
    prev_end = 0
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    references_occ = [(_.start(), _.end()) for _ in re.finditer("<ref\s*", text)]
    for t in range(len(references_occ)):
        idx, idx2 = references_occ[t]
        b2 += (text[prev_end:idx] + '\t')
        text = text[idx:]
        comp = "</ref\s*>"
        ns = references_occ[-1][1]
        if t != len(references_occ) - 1:
            ns = references_occ[t+1][0] 
        poss_end = re.search(comp, text)
        if poss_end != None:
            if poss_end.start() >= ns:
                comp = "/>"
        else:
            comp = "/>"
        curr = re.search(comp, text)
        if curr != None:
            close_idx = curr.end()
            r2 += (text[:close_idx] + '\n')
            prev_end = close_idx
        else:
            b2 += text
            prev_end = len(text)
            break
    b2 += (text[prev_end:])
    chunk_storage[doc_id] = {"i":i, "b":b + '\n' + b2, "c":c, "l":l, "r":r2 + '\n' + r, "t":title}

def get_tokens(pair):
    curr = {k: re.split("\||/|\=|_|\:|\+|,|~|\^|#|\[|\]|\(|\)|\{|\}|<|>|\!|&|;|\?|\*|%|\$|@|`|\s+", pair[1][k].lower()) for k in pair[1].keys()}
    toks_dict = {k: [tok.strip("|/=.-_:'+,~^#[](){}<>!&;?*%$@`") for tok in curr[k] if tok.strip("|/=.-_:'+,~^#[](){}<>!&;?*%$@`") not in stop_words] for k in curr}
    outy = pair[0], {k: Counter([ss.stemWord(t) for t in _v if (len(t) > 0) and (len(t) <= 40) and (t not in to_be_discarded) and (not t.isalpha())]) for k, _v in toks_dict.items()}
    return outy

def process_chunk():
    global previous_postings, inverted_index, encoded_ints, chunk_storage
    id_toks = [get_tokens((_k, _v)) for _k, _v in chunk_storage.items()]
    id_toks = sorted(id_toks, key=lambda x: x[0])
    for doc_id, toks in id_toks:
        mini_dict = {}
        for k, counts in toks.items():
            for encoded_token in counts.keys():
                token_count = int(counts[encoded_token])
                if token_count <= ENCODED_INTS_LIMIT:
                    encoded_token_count = encoded_ints[token_count]
                else:
                    encoded_token_count = encode_int(token_count)
                if mini_dict.get(encoded_token) is None:
                    gap = 0
                    if previous_postings.get(encoded_token) is not None:
                        gap = previous_postings[encoded_token]
                    curr_id = doc_id - gap
                    if curr_id <= ENCODED_INTS_LIMIT:
                        encoded_curr_id = encoded_ints[curr_id]
                    else:
                        encoded_curr_id = encode_int(curr_id)
                    mini_dict[encoded_token] = encoded_curr_id               
                mini_dict[encoded_token] += (k + encoded_token_count)
                previous_postings[encoded_token] = doc_id
        for t in mini_dict.keys():
            if inverted_index.get(t) is None:
                inverted_index[t] = (mini_dict[t])
            else:
                inverted_index[t] += ('|' + mini_dict[t])
    chunk_storage.clear()
    
def merge_sort():
    global intermediate_index_version, merge_sort_dict, inverted_index, final_index_version, secondary_index
    inverted_index_length, token_num = 0, 0
    pq = PriorityQueue()
    next_elems = set()
    for i in range(1, intermediate_index_version):
        _k = next(iter(merge_sort_dict[i].keys()))
        pq.put((_k, merge_sort_dict[i][_k], i))
        del merge_sort_dict[i][_k]
    while not pq.empty():
        curr_min_key, curr_val, curr_dict = pq.get()
        token_num += 1
        print("Outy: ", token_num, curr_min_key)
        next_elems.add(curr_dict)
        inverted_index[curr_min_key] = curr_val
        while not pq.empty():
            _k, _v, dict_num = pq.get()
            print("Inny: ", curr_min_key)
            if _k != curr_min_key:
                pq.put((_k, _v, dict_num))
                break
            next_elems.add(dict_num)
            inverted_index[curr_min_key] += ('$' + _v)
        inverted_index_length += (len(curr_min_key) + len(inverted_index[curr_min_key]))
        if inverted_index_length >= INDEX_LENGTH_LIMIT:
            secondary_index[final_index_version] = curr_min_key
            with open(f'../../results_hindi/inverted_index/index_version_{final_index_version}.pkl', 'wb') as _f:
                pickle.dump(inverted_index, _f)
            final_index_version += 1
            inverted_index.clear()
            inverted_index_length = 0
        for _n in next_elems:
            if merge_sort_dict.get(_n) is None:
                continue
            if len(merge_sort_dict[_n]) > 0:
                _k = next(iter(merge_sort_dict[_n].keys()))
                pq.put((_k, merge_sort_dict[_n][_k], _n))
                del merge_sort_dict[_n][_k]
            else:
                if os.path.isfile(f'../../results_hindi/intermediates/intermediate_version_{_n}.pkl'):
                    curr_key_count, buffer_length, curr_ptr, latest_key = 0, 0, 0, 0
                    with open(f'../../results_hindi/intermediates/intermediate_version_{_n}.pkl', 'rb') as _f:
                        curr_intermediate_dict = pickle.load(_f)
                        curr_key_count = len(curr_intermediate_dict.keys())
                        while curr_key_count > 0 and buffer_length <= BUFFER_LIMIT:
                            curr_k = next(iter(curr_intermediate_dict.keys()))
                            if curr_ptr == 0:
                                pq.put((curr_k, curr_intermediate_dict[curr_k], _n))
                            else:
                                merge_sort_dict[_n][curr_k] = curr_intermediate_dict[curr_k]
                            buffer_length += (len(curr_k) + len(curr_intermediate_dict[curr_k]))
                            del curr_intermediate_dict[curr_k]
                            curr_ptr += 1
                            curr_key_count -= 1
                    if curr_key_count == 0:
                        os.remove(f'../../results_hindi/intermediates/intermediate_version_{_n}.pkl')
                    else:
                        with open(f'../../results_hindi/intermediates/intermediate_version_{_n}.pkl', 'wb') as _f:
                            pickle.dump(curr_intermediate_dict, _f)
                else:
                    del merge_sort_dict[_n]
        next_elems.clear()
    if len(inverted_index) > 0:
        secondary_index[final_index_version] = curr_min_key
        with open(f'../../results_hindi/inverted_index/index_version_{final_index_version}.pkl', 'wb') as _f:
            pickle.dump(inverted_index, _f)
        inverted_index.clear()
    with open('../../results_hindi/inverted_index/secondary_index.pkl', 'wb') as _f:
        pickle.dump(secondary_index, _f)
    with open('../../results_hindi/stats.txt', 'w') as _f:
        _f.write(str(token_num))

st = time.time()             
for d in range(ENCODED_INTS_LIMIT+1):
    num, req = d, ""
    while num > 0:
        req += int_encoder[num % 86]
        num = num // 86
    encoded_ints[d] = req
for _i in range(1000):
    merge_sort_dict[_i] = OrderedDict()

curr_title, curr_text, curr_doc_id, to_delete = '', '', 0, set()
with BZ2File(sys.argv[1]) as xml_dump:
    parse_tree = ET.iterparse(xml_dump, events=("start", "end"))
    for event, elem in parse_tree:
        curr_tag = elem.tag
        tag_start = curr_tag.rfind('}')
        if tag_start >= 0:
            curr_tag = curr_tag[tag_start+1:]
        if event == 'start':
            if curr_tag == 'page':
                curr_doc_id += 1
                print(f'Doc {curr_doc_id} now')
        else:
            if curr_tag == 'title':
                curr_title = str(elem.text)
                titles_list.append(curr_title.lower())
                if len(titles_list) == TITLES_PER_FILE:
                    with open(f'../../results_hindi/titles/titles_part_{current_title_version}.pkl', 'wb') as ti:
                        pickle.dump(titles_list, ti)
                    current_title_version += 1
                    titles_list.clear()
            elif curr_tag == 'text':
                curr_text = str(elem.text)
                get_split_string(curr_doc_id, curr_title, curr_text)
                process_chunk()
                if curr_doc_id % ARTICLE_LIMIT == 0:
                    inverted_index = OrderedDict(sorted(inverted_index.items(), key=lambda s: s[0]))
                    for _k, _v in inverted_index.items():
                        if (intermediate_index_length + len(_k) + len(_v)) >= BUFFER_LIMIT:
                            break
                        merge_sort_dict[intermediate_index_version][_k] = _v
                        intermediate_index_length += (len(_k) + len(_v))
                        to_delete.add(_k)
                    intermediate_index_length = 0
                    for _k in to_delete:
                        del inverted_index[_k]
                    with open(f'../../results_hindi/intermediates/intermediate_version_{intermediate_index_version}.pkl', 'wb') as _f:
                        pickle.dump(inverted_index, _f)
                    intermediate_index_version += 1
                    inverted_index.clear()
                    to_delete.clear()
                    previous_postings.clear()
                curr_text, curr_title = '', ''
            elem.clear()
process_chunk()
inverted_index = OrderedDict(sorted(inverted_index.items(), key=lambda s: s[0]))
for _k, _v in inverted_index.items():
    if (intermediate_index_length + len(_k) + len(_v)) >= BUFFER_LIMIT:
        break
    merge_sort_dict[intermediate_index_version][_k] = _v
    intermediate_index_length += (len(_k) + len(_v))
    to_delete.add(_k)
intermediate_index_length = 0
for _k in to_delete:
    del inverted_index[_k]
with open(f'../../results_hindi/intermediates/intermediate_version_{intermediate_index_version}.pkl', 'wb') as _f:
    pickle.dump(inverted_index, _f)
intermediate_index_version += 1
inverted_index.clear()
previous_postings.clear()
if len(titles_list) > 0:
    with open(f'../../results_hindi/titles/titles_part_{current_title_version}.pkl', 'wb') as ti:
        pickle.dump(titles_list, ti)
        titles_list.clear()
merge_sort()
shutil.rmtree('../../results_hindi/intermediates')
e = time.time()
print(e - st)