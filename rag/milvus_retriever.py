import os
from os import path
import copy

from openai import OpenAI
from pymilvus import AnnSearchRequest
from pymilvus import WeightedRanker, RRFRanker
from pymilvus import Collection, connections
from pymilvus import connections
from pymilvus.model.sparse import BM25EmbeddingFunction
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
analyzer = build_default_analyzer(language="kr")

class milvus_retriever():
    analyzer = build_default_analyzer(language="kr")
    
    def __init__(self, openai_api_key, namespace, sparse_params, dense_params):
        self.collection = None
        self.collection_name = None
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.namespace = namespace
        
        self.bm25_ef = BM25EmbeddingFunction(milvus_retriever.analyzer)
        self._load_bm25_ef(namespace)

        self.sparse_search_params = {
            "anns_field": "bm25_vector",
            "param": {
                "metric_type": "IP",
                "params": sparse_params
            },
            "limit":5,
            "expr": f'car_type == "{namespace}"'
        }
        self.dense_search_params = {
            "anns_field": "vector",
            "param": {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": dense_params
            }, # the ratio of small vector values to be dropped during search.
            "limit":5,
            "expr": f'car_type == "{namespace}"'
        }

    def connect(self, milvus_host, milvus_port, collection_name):
        connections.connect(alias="default", host=milvus_host, port=milvus_port)
        self.collection_name = collection_name
        self.collection = Collection(name=collection_name)


    def _load_bm25_ef(self, namespace):
        bm25_dir = path.dirname(path.abspath(__file__)) + '/../vector_db/bm25'
        self.bm25_ef.load(bm25_dir + f"/bm25_{namespace}_params.json")
    def _convert_sparse(self, bm25_dict):
        return dict(map(lambda x: (x[0][1], x[1]), bm25_dict.todok().items()))
        
    def openai_embedding(self, query):
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=query
        )
        return [response.data[0].embedding]

    def bm25_embedding(self, query):
        if not isinstance(query, list):
            query = [query]
        sparse_embeddings = list(self.bm25_ef.encode_queries(query))[0]
        sparse_embeddings = [self._convert_sparse(sparse_embeddings)]
        return sparse_embeddings

    def search(self, query, rerank_weight=(0.2, 0.8), topK=5):
        sparse_params = copy.deepcopy(self.sparse_search_params)
        dense_params = copy.deepcopy(self.dense_search_params)

        sparse_params['limit'] = topK*2
        dense_params['limit'] = topK*2
        sparse_params['data'] = self.bm25_embedding(query)
        dense_params['data'] = self.openai_embedding(query)
        
        
        sparse_req = AnnSearchRequest(**sparse_params)
        dense_req = AnnSearchRequest(**dense_params)
        
        is_empty_token = len(sparse_params['data'][0]) == 0
        if is_empty_token:
            reqs =dense_params
            reqs['limit'] = int(reqs['limit']/2)
            res = self.collection.search(
                **dense_params,
                #reqs, 
                output_fields=[
                    'id', 'majorheading', 'minorheading', 'minorheading_sub_id', 'parent_doc_id', 'doc_contents', 'img_urls', 'table_img_urls', 'table_csv_urls', 'embedding_contents']
            )
            search_dict = {}
        else:
            reqs = [sparse_req, dense_req]    
            rerank = RRFRanker()#WeightedRanker(*rerank_weight)
            res = self.collection.hybrid_search(
                reqs, 
                rerank, 
                limit=topK,
                output_fields=[
                    'id', 'majorheading', 'minorheading', 'minorheading_sub_id', 'parent_doc_id', 'doc_contents', 'img_urls', 'table_img_urls', 'table_csv_urls', 'embedding_contents'
                    ]
            )
            search_dict = {}
        
        for i, result in enumerate(res[0]):
            doc_entity = result.entity
            doc = {
                'distance': result.distance,
                'id': doc_entity.get('id'),
                'majorheading': doc_entity.get('majorheading'),
                'minorheading': doc_entity.get('minorheading'),
                'minorheading_sub_id': doc_entity.get('minorheading_sub_id'),
                'parent_doc_id': doc_entity.get('parent_doc_id'),
                'doc_contents': doc_entity.get('doc_contents'),
                'img_urls': doc_entity.get('img_urls'),
                'table_img_urls': doc_entity.get('table_img_urls'),
                'table_csv_urls': doc_entity.get('table_csv_urls'),
                'embedding_contents': doc_entity.get('embedding_contents')
            }
            search_dict[i] = doc
            
        return search_dict
    
