#IRE MINI-PROJECT PHASE-2


## Dependencies needed

* nltk: For installing, execute `pip install nltk`
* PyStemmer: For installing, execute `pip install PyStemmer`
* bz2file: For installing, execute `pip install bz2file`
* ordered_set: For installing, execute `pip install ordered_set`


## Description and Execution - Indexer

* The file `indexer_english.py` contains entire implementation corresponding to the creation of inverted index for english wikipedia dump. Likewise, `indexer_hindi.py` contains implementation corresponding to the creation of inverted index for hindi wikipedia dump.

* Above files can be executed by running the commands: `python3 indexer_english.py <path_to_wiki_dump>` and `python3 indexer_hindi.py <path_to_wiki_dump>` respectively.

* Each creates 2 folders and 1 file. 

* The file created is `stats.txt` which contains information about the stats for the inverted index.

* One folder called `titles` is created, it contains document_id - title mappings. As the size is large, for loading efficiency, it has been split into multiple files of the format `titles_part_i.pkl`, where each such pickle file has a fixed number of titles (which is decoded during search).

* Another folder, called `inverted_index` is created. It contains token - posting list mappings as a dictionary. As the size is large, for loading efficiency, it has been split into multiple files of the format `index_version_i.pkl`, where each is of approximately similar size and has keys (encoded tokens) sorted in lexicographical order.

* The `inverted_index` folder has another file called `secondary_index.pkl` which contains details about the last token for each of the files above. This is useful during search so that each token's corresponding index can be found easily via binary search as all tokens are sorted.

* All of the above files and folders are created in a `results_english` / `results_hindi` folder accordingly.



## Description and Execution - search

* The file `search_english.py` / `search_hindi.py` contains implementation corresponding to search of query provided for corresponding language's wikipedia dump, using inverted index previously created.

* Above files can be executed by running the commands: `python3 search_english.py <path_to_queries_file>` / `python3 search_hindi.py <path_to_queries_file>`.

* It uses binary search and caching for faster results output, and standard TF-IDF for ranking documents (Each section of a document like title, infobox etc. is weighted differently).
