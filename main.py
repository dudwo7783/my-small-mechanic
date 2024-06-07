from typing import Union
from fastapi import FastAPI
from PIL import Image
import os
import io
import base64



from rag.car_manual_bot import car_manual_generator

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]="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"
os.environ["LANGCHAIN_PROJECT"]="my-small_mechanic"

milvus_host = 'localhost'#os.environ["MILVUS_HOST"]
milvus_port = os.environ["MILVUS_PORT"]

DB_COLLECTION_NAME = "TEST"
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
NAMESPACE_TYPE = ["IONIQ5_2024", "SANTAFE_MX5_2024", "SONATA_DN8_2024"]
NAMESPACE = NAMESPACE_TYPE[0]

CONTEXT_PATH = '/Users/yj/Kim/1.work/SKR/8.GenAI/my-small-mechanic/pdf_context'
sparse_params = {"drop_ratio_search": 0.01}
dense_params = {"ef": 100}

# text_generator = car_manual_generator(OPENAI_API_KEY, NAMESPACE, milvus_host, milvus_port, DB_COLLECTION_NAME, 10, rrk_weight=(0, 1),
#                                       score_filter=True, threshold=0.3, drop_duplicates=True, context_path=CONTEXT_PATH)

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

@app.get("/namespace/{namespace}")
def generate_car_manual_answer(namespace: str, q: str = None):
    text_generator = car_manual_generator(OPENAI_API_KEY, namespace, milvus_host, milvus_port, DB_COLLECTION_NAME, 10, rrk_weight=(0, 1),
                                      score_filter=True, threshold=0.3, drop_duplicates=True, context_path=CONTEXT_PATH)
    reduce_answer, context_answer, context_bag, docs, scores =  text_generator.generate_answer(q)
    
    pil_image_list = []
    for img_url in context_bag['image_urls']:
        img_abs_path = os.path.join(CONTEXT_PATH + '/' + img_url)
        img = Image.open(img_abs_path)
        pil_image_list.append(from_image_to_bytes(img))
    return {"query": q, "answer": reduce_answer, 'image': pil_image_list}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}