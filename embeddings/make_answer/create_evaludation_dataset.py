import os
import pandas as pd
import numpy as np
from tqdm import tqdm

import openai
from langchain_community.vectorstores import Milvus

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

from multiprocessing import Pool

num_cores = 6



# 임베딩 모델이름
company = 'OPENAI'
embedding_models = 'text-embedding-3-large'
header_include = True
table_include = True
split_yn = False
documnet_name = "2024 IONIQ5"

exp_name = f"{company}_{embedding_models}_h{header_include}_t{table_include}_s{split_yn}"
exp_dir = '/Users/yj/Kim/3.study/5.GenAI/my-small-mechanic/embeddings/experiment/result/OPENAI_text-embedding-3-large_hTrue_tTrue_sFalse'

embedding_result_path = exp_dir + '/embeddings.parquet'
df = pd.read_parquet(embedding_result_path)

df = df[df['h1']!=df['doc_contents']]
df = df[df['h2']!=df['doc_contents']]
df = df[df['h3']!=df['doc_contents']]

df = df.reset_index(drop=True).reset_index()

#NaN 처리
df[['table_contents']] = df[['table_contents']].fillna('')
df['img_urls'] = df['img_urls'].apply(lambda d: d.tolist() if d is not None else [])
df['table_img_urls'] = df['table_img_urls'].apply(lambda d: d.tolist() if d is not None else [])

llm = ChatOpenAI(api_key=os.environ['OPENAI_API_KEY'], temperature=0, model="gpt-3.5-turbo")

retriever_prompt_template = """
\n\nHuman: Here is the context information, inside <context></context> XML tags.
Please don't make questions with the contents of the table.

<chapter>{chapter}</chapter>Given the chapter name
<majorheading>{majorheading}</majorheading>Given the major heading of chapter.
<minorheading>{minorheading}</minorheading>Given the minor heading of chapter.
<context>{context}</context>Given the context information and not prior knowledge.
generate only questions based on the below query.
You are a Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \quiz/examination.
The questions should be diverse in nature \across the document.
The questions should not contain options, start with "-"
Restrict the questions to the context information provided.
Write in Korean. 

\n\nAssistant:"""

PROMPT_RETRIEVER = PromptTemplate(
    template=retriever_prompt_template,
    input_variables=["context", "num_questions_per_chunk"]
)

generation_prompt_template = """
Here is the context, inside <context></context> XML tags.

<context>
{context}
</context>
Only using the context as above, answer the following question with the rules as below:
    - Don't insert XML tag such as <context> and </context> when answering.
    - Write as much as you can
    - Be courteous and polite
    - Only answer the question if you can find the answer in the context with certainty.
    - Skip the preamble
    - Use three sentences maximum and keep the answer concise.
    - If the answer is not in the context, just say "Could not find answer in given contexts."
    - The each answers should start with "-"
    - Answer in Korean.
Question:
{question}
Answer:"""

PROMPT_GENERATION = PromptTemplate(
    template=generation_prompt_template,
    input_variables=["context", "question"]
)




chain1 = LLMChain(llm=llm,prompt=PROMPT_RETRIEVER,output_key="question",verbose=False)
chain2 = LLMChain(llm=llm,prompt=PROMPT_GENERATION,output_key="answer", verbose=False)

chain = SequentialChain(chains=[chain1,chain2],
                        input_variables=["chapter", "majorheading", "minorheading", "context","num_questions_per_chunk"],
                        output_variables=['context', 'question','answer'],verbose=False)

df['question'] = ''
df['answer'] = ''

def make_qa_from_pandas(pdf):
    for i, row in tqdm(pdf.iterrows()):
        try:
            chapter = row['h1']
            majorheading = row['h2']
            minorheading = row['h3']
            context = row['doc_contents']
            
            if len(context)<512:
                num_questions_per_chunk = 1
            elif (len(context)>=512) and (len(context)<1024):
                num_questions_per_chunk = 2
            else:
                num_questions_per_chunk = 3
            
            result = chain({'chapter': chapter, 'majorheading': majorheading, 'minorheading': minorheading, 'context': context,'num_questions_per_chunk':num_questions_per_chunk})
            question = result['question']
            answer = result['answer']
                
            df.loc[i, 'question'] = question
            df.loc[i, 'answer'] = answer
        except:
            df.loc[i, 'question'] = "error"
            df.loc[i, 'answer'] = "error"
    return df



def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_cores)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


if __name__ == "__main__":

    result = parallelize_dataframe(df, make_qa_from_pandas)
    result.to_parquet('/Users/yj/Kim/3.study/5.GenAI/my-small-mechanic/embeddings/evaluation_dataset/llm_qa_testset_gpt35_turbo.parquet', index=False)
    result.to_csv('/Users/yj/Kim/3.study/5.GenAI/my-small-mechanic/embeddings/evaluation_dataset/llm_qa_testset_gpt35_turbo.csv', index=False, encoding='cp949')