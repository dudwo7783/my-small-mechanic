
import requests
import os
from os import path
import numpy as np
import time
from IPython.display import display
from PIL import Image
from matplotlib.pyplot import imshow
import pandas as pd
import logging

from langchain_core.documents.base import Document
from langchain.prompts import (
    ChatPromptTemplate, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
import asyncio
import nest_asyncio

import pandas as pd
import numpy as np

from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


from milvus_retriever import milvus_retriever
nest_asyncio.apply()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(u'%(asctime)s [%(levelname)s] %(message)s')

# log 출력
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

    
class car_manual_generator():

    def __init__(self, openai_api_key, namespace, milvus_host, milvus_port, db_collection_name, topK, llm_model="gpt-3.5-turbo", rrk_weight=(0.3,0.7),
                 score_filter=True, threshold=0.3, drop_duplicates=False, pandas_llm_model="gpt-3.5-turbo", reduce_model="gpt-3.5-turbo",
                 map_text_model="gpt-3.5-turbo", context_path='../pdf_context'):
        self.openai_api_key = openai_api_key
        self.namespace = namespace
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.db_collection_name = db_collection_name
        self.topK = topK

        self.retriever = self._get_retriever()
        self.llm = ChatOpenAI(api_key=openai_api_key, 
                              temperature=0, 
                              model=llm_model, 
                              streaming=True, 
                              callbacks=[StreamingStdOutCallbackHandler()])
        self.rrk_weight = rrk_weight
        self.score_filter = score_filter
        self.threshold = threshold
        self.drop_duplicates = drop_duplicates
        self.pandas_llm_model = pandas_llm_model
        self.reduce_model = reduce_model
        self.map_text_model = map_text_model
        self.context_path = context_path



        prompt_dir = path.dirname(path.abspath(__file__)) + '/prompt'
        with open(prompt_dir + '/qa_map_template.txt', 'r') as f:
            self.qa_map_prompt = f.read()
            
        with open(prompt_dir + '/qa_reduce_template.txt', 'r') as f:
            self.qa_reduce_prompt = f.read()

    def _get_retriever(self):
        sparse_params = {"drop_ratio_search": 0.01}
        dense_params = {"ef": 100}
        retriever = milvus_retriever(self.openai_api_key, self.namespace, sparse_params, dense_params)
        retriever.connect(self.milvus_host, self.milvus_port, self.db_collection_name)
        return retriever

    def _remove_duplicates(self, lst):
        seen = set()
        result = []
        for k, item in lst.items():
            if item['doc_id'] not in seen:
                seen.add(item['doc_id'])
                result.append(item)
        return result
        
    def get_context(self, query):
        retriever_result = self.retriever.search(query, rerank_weight=self.rrk_weight, topK=self.topK)
        child_text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=512,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        pairs = list(map(lambda k: {"text":k[0], "text_pair":retriever_result[k[1]]['embedding_contents']}, zip([query]*len(retriever_result), retriever_result.keys())))
        child_chunk = pd.DataFrame(pairs)['text_pair'].apply(lambda x: list(map(lambda x: x.page_content, child_text_splitter.create_documents([x])))).explode()
        pairs = list(map(lambda k: {"text":k[0], "text_pair":k[1]}, zip([query]*len(child_chunk), child_chunk)))
                
        def rerank(payload):
            API_URL = "https://api-inference.huggingface.co/models/Dongjin-kr/ko-reranker"
            headers = {"Authorization": "Bearer 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"}

            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()
            
     
        logger.info("Start Rerank")
        for i in range(0,3):   
            output = rerank({
                "inputs": pairs,
            })
            if 'error' not in output:
                break
            else:
                logger.info(f"Rerank retry {i+1}..")
                time.sleep(10)
        logger.info("End Rerank")

        scores = list(map(lambda x: x[0]['score'], output))
        score_df = child_chunk.iloc[np.argsort(scores)][::-1].to_frame()
        score_df['score'] = np.array(scores)[np.argsort(scores)[::-1]]
        score_df = score_df.reset_index().drop_duplicates(subset=['index'], keep='first')
        score_idx = score_df['index'].values
        score = score_df.score


        result = []
        for si in score_idx:
            result.append(retriever_result[si])
        
        if self.score_filter:
            relevent_rrk_doc_idx = np.where(score>=self.threshold)[0]
            result = np.array(result)[relevent_rrk_doc_idx].tolist()
            score = score.reset_index(drop=True)[relevent_rrk_doc_idx]

        if self.drop_duplicates:
            result_df = pd.DataFrame(result)
            result_df['score'] = score
            result_df = result_df.drop_duplicates('parent_doc_id', keep='first')
            score = result_df['score']
            result = []
            for i, row in result_df.iterrows():
                result.append(row.to_dict())
        
        return result, score
    
    

    def context_parser(self, docs):
        
        page_contents = '\n\n'.join([d['doc_contents'] for d in docs])
        img_urls = []
        tbl_img_urls = []
        tbl_csv_urls = []
        # tbls = []
        
        for d in docs:
            img_urls.extend(d['img_urls'])
            tbl_img_urls.extend(d['table_img_urls'])
            tbl_csv_urls.extend(d['table_csv_urls'])
            # tbls.extend(d.metadata['tbl_contents'])
            # doc_ids.append(d.metadata['doc_id'])

        parsed_result = {
            'page_contents': page_contents,
            'img_urls': img_urls,
            'table_img_urls': tbl_img_urls,
            'table_csv_urls': tbl_csv_urls,
            # 'doc_id': doc_ids
        }
        return parsed_result
    
    def put_context_to_bag(self, items, bag_size):
        bags = {}
        bag_index = 0
        current_bag = []
        current_size = 0

        for i, item in enumerate(items):
            if item > bag_size:
                bags[bag_index] = [i]
                bag_index += 1
            else:
                if current_size + item <= bag_size:
                    current_bag.append(i)
                    current_size += item
                else:
                    bags[bag_index] = current_bag
                    bag_index += 1
                    current_bag = [i]
                    current_size = item

        if current_bag:
            bags[bag_index] = current_bag

        return bags


    def create_context_bag(self, docs, bag_size=4000): 
        image_urls = []
        table_image_urls = []
        table_csv_urls = []

        context_len = list(map(lambda x: len(x['doc_contents']), docs))
        total_context_len = sum(context_len)

        list(map(lambda x: image_urls.extend(x['img_urls']), docs))
        list(map(lambda x: table_image_urls.extend(x['table_img_urls']), docs))
        list(map(lambda x: table_csv_urls.extend(x['table_csv_urls']), docs))

        context_bag = {}
        context_bag_map = self.put_context_to_bag(context_len, bag_size)
        for k, v in context_bag_map.items():
            contexts_in_bag = np.array(docs)[context_bag_map[k]]
            context_bag[k] = '\n\n'.join(list(map(lambda x: x['doc_contents'], contexts_in_bag)))
            
        context_bag = {
            'total_context_size': total_context_len,
            'image_urls': image_urls,
            'table_image_urls': table_image_urls,
            'table_csv_urls': table_csv_urls,
            'text_context_bag': context_bag
        }
        return context_bag


    def reduce_context_string(self, map_result):
        processed_results = []
        for key, value in map_result.items():
            processed_results.append(f"{key}: {value}")
        return "\n".join(processed_results)

    async def map_text_context(self, context_bag, query):
        map_llm = ChatOpenAI(api_key=self.openai_api_key, temperature=0, model=self.map_text_model) 
            
        map_context_chains = {}
        for k, context in context_bag['text_context_bag'].items():
            qa_map_template = ChatPromptTemplate.from_template(self.qa_map_prompt)
            chain = qa_map_template.partial(context=context) | map_llm | StrOutputParser()
            map_context_chains[f'context{k}'] = chain
            
        map_text_context_chain = RunnableParallel(**map_context_chains)
            
        map_text_context_answer = await map_text_context_chain.ainvoke({"query": query})
        return map_text_context_answer    
    
    # async -> 비동기 함수로 변경
    async def invoke_table_context(self, df, query):
        agent = create_pandas_dataframe_agent(
            ChatOpenAI(api_key=self.openai_api_key, temperature=0, model=self.pandas_llm_model, streaming=True),
            df, verbose=False, agent_type=AgentType.OPENAI_FUNCTIONS)
        
        response = await agent.ainvoke(query + '\n 당신이 알고 있는 사전지식에서 대답하지말고 주어진 표에서 대답하세요. 찾을 수 없다면 모른다고 대답하세요.')
        return response

    async def map_table_context(self, dfs, query):
        async_agent_tasks = list(map(lambda df: asyncio.create_task(self.invoke_table_context(df, query)), dfs))
        # asyncio.wait : 함수 호출을 알아서 스케줄링하여 비동기로 호출
        output = await asyncio.gather(*async_agent_tasks)
        return output

    async def map_context(self, context_bag, query):
        
        async_map_task = [asyncio.create_task(self.map_text_context(context_bag, query))]
        if len(context_bag['table_csv_urls'])>0:
            context_root_dir = self.context_path
            dfs = list(map(lambda x: pd.read_csv(context_root_dir + '/' + x), context_bag['table_csv_urls']))
            async_map_task.append(asyncio.create_task(self.map_table_context(dfs, query)))
        output = await asyncio.gather(*async_map_task)
        return output
    
    def generate_answer(self, query):
        logger.info("Start searching DB")
        docs, scores = self.get_context(query)
        context_bag = self.create_context_bag(docs)
        logger.info("End searching DB")

        logger.info("Start map query")
        async_map_context_result = asyncio.run(self.map_context(context_bag, query))
        logger.info("End map query")

        context_answer = list(map(lambda x: x[1], async_map_context_result[0].items()))
        if len(context_bag['table_csv_urls'])>0:
            context_table_retriever_result = list(map(lambda x: x['output'], async_map_context_result[1]))
            context_answer = context_answer + context_table_retriever_result
        context_answer = dict(map(lambda x: (f"context{x[0]}", x[1]), zip(range(len(context_answer)), context_answer)))

        reduce_model = ChatOpenAI(api_key=self.openai_api_key, 
                              temperature=0, 
                              model=self.reduce_model,
                              streaming=True,
                              callbacks=[StreamingStdOutCallbackHandler()]) 
        qa_reduce_template = ChatPromptTemplate.from_template(self.qa_reduce_prompt)
        reduce_chain = qa_reduce_template | reduce_model | StrOutputParser()
        logger.info("Start reducing Context")
        reduce_answer = reduce_chain.invoke({"query": query, "context":context_answer})
        logger.info("\nEnd reducing Context")
        
        context_root_dir = self.context_path
        for img_path in context_bag['image_urls']:
            # TODO: 벡터 DB 구축시 이미지 URL 변경필요
            img_path = os.path.join(context_root_dir, img_path)
            img = Image.open(img_path)
            display(img)
            # imshow(np.asarray(img))

            # img.show()
            # display(img)
            
        for img_path in context_bag['table_image_urls']:
            # TODO: 벡터 DB 구축시 이미지 URL 변경필요
            # img_path = img_path.split('/')[-1]
            img_path = os.path.join(context_root_dir, img_path)
            img = Image.open(img_path)
            display(img)
            # imshow(np.asarray(img))

            # img.show()
            # display(img)   
        return reduce_answer, context_answer, context_bag, docs, scores
    