

import random
import time
import streamlit as st
from code_rag.rag import QA
import sqlite3
import json

from langchain.agents import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains import LLMChain
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory

import httpx
import asyncio
from httpx import TimeoutException

async def get_streaming_response(namespace, query):
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream('GET', 'http://localhost:8000/aget_car_information/', params={'namespace': namespace, 'query': query}) as response:
            # TODO: ver 1
            # TODO: json 분리 확인 1 response['text']
            # TODO: 안되면 아래처럼
            async for chunk in response.aiter_text():
                yield chunk

class ChatBot:
    def __init__(self):
        self.messages = []
        self.type=0
        self.id=1
        self.answer = []
        self.set_num = -999
    # 글자 타이핑 되도록 이펙트
    def stream_data(self, text):
        st.toast("답변 생성중입니다......")
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.2)

    # Button 클릭 시 작동
    def display_question_buttons(self, answer_type):
        conn = sqlite3.connect('button.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM button WHERE id=?", (self.id,))
        result = cursor.fetchone()
        response = json.loads(result[2])
        # print(result)
        if result[4] == 0:
            with st.chat_message("assistant"):
                for index, res in enumerate(response):
                    print(res, index)
                    ind = str(self.id) + '_' + str(index)
                    if st.button(res, key=f"question_{ind}", type="primary"):
                        cursor.execute("SELECT * FROM button WHERE question=?", (res,))
                        self.answer = cursor.fetchone()
                        if self.answer[4]==2:
                            self.messages.append({"role": "user", "content": res})
                            self.type = 1
                        else:
                            # TODO: self.id정의를 끝에서 하되 여기서 누른것의 id를 self.id로 하는 것을 마지막에 추가해줘야함 그럼 breakㅠ필요 없음
                            st.session_state.clicked = True
                            text = response
                            assis_res = {"text": [text, res], 'image': None}
                            self.messages.append({"role": "assistant", "content": assis_res, "answer_type": answer_type})
                            self.messages.append({"role": "user", "content": res})
                else:
                    if self.answer:
                        self.set_num = self.answer[0]
            if self.set_num != -999:
                self.id = self.set_num
            # st.rerun()
            if st.session_state.clicked:
                st.session_state.clicked = False
                st.rerun()
        else:
            self.type = 1
            text = json.loads(result[2])[0]
            image_paths = json.loads(result[3])
            res = {"text": text, "image": image_paths}
            question = result[1]
            answer_type='image'
            # self.messages.append({"role": "user", "content": question})
            self.messages.append({"role": "assistant", "content": res, "answer_type": answer_type})
            with st.chat_message("assistant"):
                
                st.write_stream(response)
                with st.expander("Click to view images"):
                    if len(image_paths) != 0:
                        cols = st.columns(len(image_paths))
                        for i, image_path in enumerate(image_paths):
                            with cols[i]:
                                st.image(image_path)
            
            asyncio.run(self.handle_user_input(answer_type))
            # Display a message while waiting for the response

    def display_chat_history(self):
        for message in self.messages:
            with st.chat_message(message["role"]):
                content = message["content"]
                if message["role"]=='assistant':
                    # print("displayChat: content")
                    # print(content)
                    if message['answer_type']=='image':
                        st.markdown(content["text"])
                    else:
                        for index, res in enumerate(content['text'][0]):
                            if res != content['text'][1]:
                                st.button(res)
                            else:
                                st.button(res, type="primary")
                else:
                    st.markdown(content)
                
                if message["role"]=='assistant' and content["image"] != None and message["answer_type"] == "image":
                    image_paths = content["image"]
                    if len(image_paths) != 0:
                        with st.expander("Click to view images"):
                            cols = st.columns(len(image_paths))
                            for i, image_path in enumerate(image_paths):
                                # print("#1 Image Path")
                                # print(image_path)
                                with cols[i]:
                                    st.image(image_path)

    async def handle_user_input(self, answer_type):
        user_input = st.chat_input("질문할 내용을 적어주세요: ", key="user_input")
        if user_input:
            
            st.chat_message("user").write(user_input)
            st.toast("서칭 중......")
            self.messages.append({"role": "user", "content": user_input})
            st.toast('Yeaaaaaaaaah',  icon='🎉')
            with st.chat_message("assistant"):
                
                # st.write_stream(self.stream_data(assistant_response["text"]))
                # with st.expander("Click to view images"):
                #     if assistant_response["image"]!= None:
                #         image_paths = assistant_response["image"]
                #         if len(image_paths) != 0:
                #             cols = st.columns(len(image_paths))
                #             for i, image_path in enumerate(image_paths):
                #                 with cols[i]:
                #                     st.image(image_path)
                
                with st.spinner("Waiting for response..."):
                    try:
                        container = st.empty()
                        # Buffer to store the incoming chunks
                        buffer = b""
                        boundary = b"my-custom-boundary"
                        image_paths = ''
                        
                        answer = ''
                        

                        # Get the streaming response from FastAPI
                        async for chunk in get_streaming_response('SONATA_DN8_2024', user_input):
                            buffer += chunk.encode('utf-8')  # chunk를 바이트로 변환
                            parts = buffer.split(boundary)
                            # buffer = parts.pop()
                            # print(buffer)
                            for part in parts:
                                part_data = part.split(b"\r\n\r\n")

                                if len(part_data) >= 2:
                                    header, data = part_data[0], part_data[1]
                                    if b"Content-Type: text/event-stream" in header:
                                        # print('after if')
                                        # print(data)
                                        # logger.info(data.decode("utf-8"))
                                        # await asyncio.sleep(1)
                                        token = data.decode("utf-8")  # .rstrip()
                                        print(token)
                                        answer += token
                                        
                                    elif b"Content-Type: text/plain" in header:
                                        # image_data.append(data.strip())
                                        image_paths = data.decode('utf-8')  # 이미지 경로를 문자열로 변환
                            container.markdown(answer)# unsafe_allow_html=True
                        else:
                            with container.expander("Click to view images"):
                                if len(image_paths) != 0:
                                    cols = st.columns(len(image_paths))
                                    for i, image_path in enumerate(image_paths):
                                        with cols[i]:
                                            st.image(image_path)
                            assistant_response = {"text": answer, "image": image_paths}
                            self.messages.append({"role": "assistant", "content": assistant_response, "answer_type": answer_type})
                    except TimeoutException:
                        pass

    def side_bar(self):
        with st.sidebar:
            car_model = st.selectbox(
                "차정",
                ("그랜저", "아이오닉 5", "싼타페", "쏘나타")
            )
            st.session_state.car_model = car_model
            for ind, chat in enumerate(['./messages_1.txt', './messages_2.txt', './messages_3.txt'], 1):
                with open(chat, "r") as file:
                    # 파일 내용 읽기
                    content = file.read()
                    # JSON 형식으로 변환
                    temp = json.loads(content)
                first_text = temp[-1]['content']['text'][:10]
                if st.button(f'Chat_{ind}: {first_text}', key=f"side_bar_{ind}", type="secondary"):
                    st.session_state.clicked = True
                    self.messages = temp
            if st.session_state.clicked:
                st.session_state.clicked = False
                self.type = 1
                st.rerun()


    def run(self):
        print("START")
        self.side_bar()
        # with st.spinner("Loading..."):
        #     time.sleep(5)
        # st.success("Done!")
        st.title(f"My Small Javis ver.{st.session_state.car_model}")
        # 첫 클릭 후에도 first, third in 둘다 발생
        # TODO: 버튼을 눌렀을 경우 바로 message에 append 되지 않음
        print(self.messages)
        if len(self.messages) == 0:
            print("First if IN")
            answer_type = "button"
            self.display_question_buttons(answer_type)

        print("First if AFTER")
        self.display_chat_history()
        if len(self.messages) > 0:
            # print(1)
            # print("Second if IN")
            last_message = self.messages[-1]
            answer_type = last_message.get("answer_type", "image")
            if self.type==0:
                # print(2)
                print("Third if IN")
                answer_type = "button"
                self.display_question_buttons(answer_type)
            else:
                # print(3)
                print("Forth if IN")
                answer_type="image"
                asyncio.run(self.handle_user_input(answer_type))
        print("END")

if "chat_bot" not in st.session_state:
    st.session_state.clicked = False
    st.session_state.personal_id=-999
    st.session_state.car_model = '아이오닉 5'

    st.session_state.chat_bot = ChatBot()

st.session_state.chat_bot.run()






            # with open("./messages.txt", "w") as file:
            #     file.write(json.dumps(self.messages))