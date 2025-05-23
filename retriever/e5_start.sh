index_file=/root/Changling/retriever/local_wiki/e5_Flat.index
corpus_file=/root/Changling/retriever/local_wiki/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python /root/Changling/retriever/retrieval_server.py --index_path $index_file --corpus_path $corpus_file --topk 3 --retriever_name $retriever_name --retriever_model $retriever_path --faiss_gpu