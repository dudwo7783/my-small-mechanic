from typing import Union
from fastapi import FastAPI
from PIL import Image
import os
import io
from io import BytesIO
import uvicorn
import asyncio
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
os.environ['MILVUS_PORT'] = '19530'

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
MILVUS_HOST = 'localhost'#os.environ["MILVUS_HOST"]
MILVUS_PORT = os.environ["MILVUS_PORT"]
DB_COLLECTION_NAME = "TEST"
CONTEXT_PATH = '/Users/yj/Kim/1.work/SKR/8.GenAI/my-small-mechanic/pdf_context'

DEVICE='mps'
RERANKER = pipeline("text-classification", model="Dongjin-kr/ko-reranker", device=DEVICE)

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


@app.get("/aget_car_information/")
async def agenerate_car_manual_answer(namespace: str, query: str):
    stream_it = AsyncIteratorCallbackHandler()
    text_generator = car_manual_generator(OPENAI_API_KEY, namespace, MILVUS_HOST, MILVUS_PORT, DB_COLLECTION_NAME, 10, rrk_weight=(0.3, 0.7),
                                score_filter=True, threshold=0.3, drop_duplicates=True, context_path=CONTEXT_PATH, reranker=RERANKER)
    
    boundary = "my-custom-boundary"    
    async def generate_response():
        
        task = asyncio.create_task(text_generator._agenerate_answer(query, stream_it))
        async for chunk in stream_it.aiter():
            yield (f"{boundary}\r\n"
                   f"Content-Type: text/event-stream\r\n\r\n"
                   f"{chunk}").encode("utf-8")
        if not task.done():
            await task
        # get the return value from the wrapped coroutine
        answer, context_bag = task.result()
        print(context_bag)
        
        if (len(context_bag['image_urls'])!=0) or (len(context_bag['table_image_urls'])!=0):
            image = str(context_bag['image_urls'] + context_bag['table_image_urls'])
        # image = str(['./image/너구리.jpg'])
            yield (f"{boundary}\r\n"
                    f"Content-Type: text/plain\r\n\r\n"
                    f"{image}").encode("utf-8")
    

    # 응답 반환
    return StreamingResponse(generate_response(), media_type=f"multipart/form-data; boundary={boundary}")

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

if __name__ == "__main__" :
	uvicorn.run("main:app", reload=True)