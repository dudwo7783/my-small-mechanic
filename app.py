# import streamlit as st
# import httpx
# import asyncio
# from httpx import TimeoutException


# async def get_streaming_response(namespace, query):
#     async with httpx.AsyncClient(timeout=None) as client:
#         async with client.stream('GET', 'http://localhost:8000/aget_car_information/', params={'namespace':namespace,'query': query}) as response:
#             async for chunk in response.aiter_text():
#                 yield chunk

# async def process_query(namespace, query):
#     # Create a container for the streaming response
#     response_container = st.empty()

#     # Display a message while waiting for the response
#     with st.spinner("Waiting for response..."):
#         try:
#             # Buffer to store the incoming chunks
#             buffer = ""

#             # Get the streaming response from FastAPI
#             async for chunk in get_streaming_response(namespace, query):
#                 # Append the chunk to the buffer
#                 buffer += chunk
#                 response_container.markdown(buffer, unsafe_allow_html=True)

#         except TimeoutException:
#             pass

# def main():
#     st.title("Car Manual Generator")

#     # Get the query from the user
#     namespace = st.text_input("차량 ID를 입력하세요.")
    
#     query = st.text_input("Enter your question about the car manual:")

#     if st.button("Submit"):
#         if query and namespace:
#             # Process the query when the button is clicked
#             asyncio.run(process_query(namespace, query))
#         else:
#             st.warning("Please enter a question.")

# if __name__ == '__main__':
#     main()

import streamlit as st
import time
import httpx
import asyncio
from httpx import TimeoutException
from PIL import Image
from io import BytesIO
import logging
# 1. logger 생성
logger = logging.getLogger("main")

# 2. logger 레벨 설정
logger.setLevel(logging.DEBUG)
# 또는
logging.basicConfig(level=logging.DEBUG)

# 3. formatting 설정
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_hander = logging.StreamHandler()
stream_hander.setFormatter(formatter)
logger.addHandler(stream_hander)


async def get_streaming_response(namespace, query):
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream('GET', 'http://localhost:8000/aget_car_information/', params={'namespace': namespace, 'query': query}) as response:
            async for chunk in response.aiter_bytes():
                yield chunk

async def process_query(namespace, query):
    # Create containers for the streaming response and images
    response_container = st.empty()
    image_container = st.container()
    answer = ""

    # Display a message while waiting for the response
    with st.spinner("Waiting for response..."):
        try:
            # Buffer to store the incoming chunks
            buffer = b""
            boundary = b"my-custom-boundary"
            image_data = []

            # Get the streaming response from FastAPI
            async for chunk in get_streaming_response(namespace, query):
                
                buffer += chunk
                parts = buffer.split(boundary)
                buffer = parts.pop()

                for part in parts:
                    part_data = part.split(b"\r\n\r\n")
                    if len(part_data) >= 2:

                        header, data = part_data[0], part_data[1]
                        if b"Content-Type: text/plain" in header:
                            # logger.info(data.decode("utf-8"))
                            # await asyncio.sleep(1)
                            token = data.decode("utf-8")#.rstrip()
                            answer = answer + token
                            response_container.markdown(answer, unsafe_allow_html=True)
                        elif b"Content-Type: image/jpeg" in header:
                            image_data.append(data.strip())

            # Display the images
            for image_bytes in list(set(image_data)):
                image = Image.open(BytesIO(image_bytes))
                image_container.image(image, use_column_width=True)

        except TimeoutException:
            st.error("Request timed out. Please try again.")

def main():
    st.title("Car Manual Generator")

    # Get the query from the user
    namespace = st.text_input("차량 ID를 입력하세요.")
    query = st.text_input("Enter your question about the car manual:")

    if st.button("Submit"):
        if query and namespace:
            # Process the query when the button is clicked
            asyncio.run(process_query(namespace, query))
        else:
            st.warning("Please enter a question and namespace.")

if __name__ == '__main__':
    main()