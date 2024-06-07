
import os

from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from pymilvus.model.sparse import BM25EmbeddingFunction
from openai import OpenAI
from konlpy.tag import Mecab
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=OPENAI_API_KEY)


def extract_table_token(_df):
    df = _df.copy()
    mecab = Mecab()

    def pos_tag(x):
        pos_tag = mecab.pos('\n'.join(x))
        pos_tag = set(pos_tag)
        pos_tag = list(filter(lambda x: x[1] in ['NNP', 'NNG', 'MAG', 'SL'],pos_tag))
        pos_tag = list(map(lambda x: x[0], pos_tag))
        pos_tag_str = ','.join(pos_tag)
        return pos_tag_str
    df['table_token'] = df['table_contents'].apply(pos_tag)
    df['embedding_contents'] = df['h3'] + '\n' + df['child_chunk'] + '\n' + df['table_token']
  
    df['embedding_contents'] = df['h2'] + '\n\n' + df['embedding_contents']
    
    return df
    


def bm25_embedding(corpus, bm25_param_path, train=True):
    analyzer = build_default_analyzer(language="kr")
    bm25_ef = BM25EmbeddingFunction(analyzer)
    if train:
        bm25_ef.fit(corpus)
        bm25_ef.save(bm25_param_path)
    else:
        bm25_ef.load(bm25_param_path)
    embedding_result = bm25_ef.encode_documents(corpus)
    def convert_sparse(bm25_dict):
        return dict(map(lambda x: (x[0][1], x[1]), bm25_dict.todok().items()))
    
    embedding_result = list(embedding_result)
    embedding_result = list(map(lambda x: convert_sparse(x), embedding_result))
    
    
    return embedding_result
    
    
def openai_embeddings(corpus):
    
    def get_embedding(text):
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        # print(index)
        return response.data[0].embedding
    
    embedding_result = pd.Series(corpus).progress_apply(lambda x: get_embedding(x)).values
    
    # embedding_result = list(map(lambda doc: get_embedding(doc), corpus))
    
    return embedding_result