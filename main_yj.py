from typing import Union
from fastapi import FastAPI
from PIL import Image
import os
import io
from io import BytesIO
import uvicorn

import base64

from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from fastapi.responses import StreamingResponse, Response
from transformers import pipeline
from rag.car_manual_bot import car_manual_generator

from aiohttp import MultipartWriter, web

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]="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"
os.environ["LANGCHAIN_PROJECT"]="my-small_mechanic"
os.environ['OPENAI_API_KEY'] = 'apikey'
os.environ['MILVUS_PORT'] = '19530'

milvus_host = 'localhost'#os.environ["MILVUS_HOST"]
milvus_port = os.environ["MILVUS_PORT"]

DB_COLLECTION_NAME = "TEST"
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
NAMESPACE_TYPE = ["IONIQ5_2024", "SANTAFE_MX5_2024", "SONATA_DN8_2024"]
NAMESPACE = NAMESPACE_TYPE[0]

CONTEXT_PATH = '/Users/yj/Kim/1.work/SKR/8.GenAI/my-small-mechanic/pdf_context'
sparse_params = {"drop_ratio_search": 0.01}
dense_params = {"ef": 100}
device='cpu'
reranker = pipeline("text-classification", model="Dongjin-kr/ko-reranker", device=device)

text_generator = car_manual_generator(OPENAI_API_KEY, NAMESPACE, milvus_host, milvus_port, DB_COLLECTION_NAME, 10, rrk_weight=(0, 1),
                                score_filter=True, threshold=0.3, drop_duplicates=True, context_path=CONTEXT_PATH, reranker=reranker)

app = FastAPI()

def from_image_to_bytes(img):
    """
    pillow image 객체를 bytes로 변환
    """
    # Pillow 이미지 객체를 Bytes로 변환
    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format=img.format)
    imgByteArr = imgByteArr.getvalue()
    # Base64로 Bytes를 인코딩
    encoded = base64.b64encode(imgByteArr)
    # Base64로 ascii로 디코딩
    decoded = encoded.decode('ascii')
    return decoded

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/get_car_information/")
def generate_car_manual_answer(q: str = None):
    
    reduce_answer, context_answer, context_bag, docs, scores =  text_generator.generate_answer(q)
    
    pil_image_list = []
    for img_url in context_bag['image_urls']:
        img_abs_path = os.path.join(CONTEXT_PATH + '/' + img_url)
        img = Image.open(img_abs_path)
        pil_image_list.append(from_image_to_bytes(img))
    return {"query": q, "answer": reduce_answer, 'image': pil_image_list}

# @app.get("/aget_car_information/")
# async def agenerate_car_manual_answer(query: str):

#     stream_it = AsyncIteratorCallbackHandler()
#     # TODO: ver 2
#     # reduce_answer_iter = text_generator.agenerate_answer(query, stream_it) + '斷./image/너구리.jpg'
#     # TODO: ver 1
#     reduce_answer, context_answer, context_bag, docs, scores = text_generator.agenerate_answer(query, stream_it)

#     data = {
#         'text': reduce_answer,
#         'image': context_bag['image_urls']
#     }

#     # ver 2
#     # return StreamingResponse(reduce_answer_iter, media_type="text/event-stream")
#     # ver 1
#     print(data)
#     return StreamingResponse(data, media_type="application/json")

@app.get("/aget_car_information/")
async def agenerate_car_manual_answer(namespace: str, query: str):
    stream_it = AsyncIteratorCallbackHandler()
    text_generator = car_manual_generator(OPENAI_API_KEY, namespace, milvus_host, milvus_port, DB_COLLECTION_NAME, 10, rrk_weight=(0.3, 0.7),
                                          score_filter=True, threshold=0.3, drop_duplicates=True, context_path=CONTEXT_PATH, reranker=reranker)
    
    async def generate_response():
        async for chunk, context_bag in text_generator.agenerate_answer(query, stream_it):
            yield chunk, context_bag['image_urls']

    # 멀티파트 응답 생성
    boundary = "my-custom-boundary"
    
    async def iterfile():
        async for chunk, context_bag in generate_response():
            print(chunk)
            yield (f"{boundary}\r\n"
                   f"Content-Type: text/event-stream\r\n\r\n"
                   f"{chunk}").encode("utf-8")
        if len(context_bag)!=0:
            print(context_bag)
            image = str(context_bag)
            yield (f"{boundary}\r\n"
                    f"Content-Type: text/plain\r\n\r\n"
                    f"{image}").encode("utf-8")

    # 응답 반환
    return StreamingResponse(iterfile(), media_type=f"multipart/form-data; boundary={boundary}")

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

if __name__ == "__main__" :
	uvicorn.run("main:app", reload=True)
