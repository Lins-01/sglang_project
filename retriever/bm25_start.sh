index_file=/root/Changling/retriever/sparse/cache/bm25
corpus_file=/root/Changling/retriever/local_wiki/wiki-18.jsonl
retriever_name=bm25

python /root/Changling/retriever/retrieval_server.py --index_path $index_file --corpus_path $corpus_file --topk 3 --retriever_name $retriever_name
