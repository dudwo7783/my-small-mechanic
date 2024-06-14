# -*- coding: utf-8 -*-
import sqlite3
import json

conn = sqlite3.connect("button.db")
cur = conn.cursor()
conn.execute('''CREATE TABLE IF NOT EXISTS button
             (id INTEGER PRIMARY KEY, question TEXT, response TEXT, image TEXT, type int)''')

sample_questions = [
    {
        "question": "start", 
        "index": 1,
        "response": {
            "text": [
                "시동 및 도어",
                "장치",
                "주유/충전",
                "경고등",
                "직접입력"
                ],
            "image": [],
            "type": 0,
        }
    },
    {
        "question": "시동 및 도어", 
        "index": 1,
        "response": {
            "text": [
                "시동걸기",
                "스마트키 원격시동",
                "차 밖에서 문열기",
                "차 안에서 문열기"
                ],
            "image": [],
            "type": 0,
        }
    },
    {
        "question": "시동걸기", 
        "index": 2,
        "response": {
            "text": [
                '기어를 P(주차) 상태에 둔 후 브레이크 페달을 밟으면서 시동 버튼을 누르세요',
                ],
            "image": [
                './image/시동걸기.png'
                ],
            "type": 1,
        }
    },
    {
        "question": "스마트키 원격시동", 
        "index": 3,
        "response": {
            "text": [
                '차량과 10m 이내 거리에서 스마트키의 도어잠금 버튼을 누른 후 4초 이내 하단의 원격시동 버튼을 길게 누르세요\n\n*시동이 걸리면 비상 경고등이 깜빡입니다\n*원격 시동을 끄려면 원격시동 버튼을 다시 누르세요'
                ],
            "image": [
                './image/스마트키 원격시동.png'
                ],
            "type": 1,
        }
    },
    {
        "question": "차 밖에서 문열기", 
        "index": 4,
        "response": {
            "text": [
                '스마트 키를 휴대하고 앞좌석 도어핸들 쪽으로 1m 이내 접근시, 자동으로 잠금해제되면서 바깥쪽 도어 핸들이 튀어나옵니다.\n\n*\'접근 시 잠금 해제\'가 설정된 경우, 경고등이 2회 깜빡거리고 확인음이 2회 울립니다'
                ],
            "image": [
                './image/차 밖에서 문열기.png'
                ],
            "type": 1,
        }
    },
    {
        "question": "차 안에서 문열기", 
        "index": 5,
        "response": {
            "text": [
                '도어 핸들을 잡아당기면 잠금해제와 동시에 도어가 열립니다\n\n* 스마트키가 차 안에 있을 경우, 앞 좌석 도어를 열고 도어 잠금버튼이나 중앙도어 잠금버튼을 눌러도 도어가 잠기지 않습니다.'
                ],
            "image": [
                './image/차 안에서 문열기.png'
                ],
            "type": 1,
        }
    },
    {
        "question": "장치", 
        "index": 6,
        "response": {
            "text": [
                "변속 다이얼",
                "후면 트렁크",
                "전면 트렁크"
                ],
            "image": [],
            "type": 0,
        }
    },
    {
        "question": "변속 다이얼", 
        "index": 7,
        "response": {
            "text": [
                '브레이크를 밟고 아래 이미지와 같이 변속다이얼을 각 위치로 돌려 변속할 수 있습니다.\n\n*변속단 표시등이 원하는 변속단으로 표시되었는지 반드시 확인해 주세요',
                ],
            "image": [
                './image/변속 다이얼.png'
                ],
            "type": 1,
        }
    },
    {
        "question": "후면 트렁크", 
        "index": 8,
        "response": {
            "text": [
                '차량을 주차상태로 놓고, 아래 방법 중 하나를 선택하여 트렁크를 열어주세요.\n1. 스마트키의 테일게이트 작동 버튼을 길게 누르세요.\n2. 실내에서 열림/닫힘 버튼을 누르세요\n3. 실외에서는 스마트키를 휴대하고 파워 테일게이트 핸들 스위치를 눌러 열고, 테일게이트 닫힘 버튼을 누르면 경고등과 함께 닫힙니다.\n\n*파워 테일게이트 작동 중에 버튼을 짧게 누르면 작동이 멈춥니다.\n테일게이트를 닫을 때는 버튼을 계속 누르세요'
                ],
            "image": [
                './image/후면 트렁크.png'
                ],
            "type": 1,
        }
    },
    {
        "question": "전면 트렁크", 
        "index": 9,
        "response": {
            "text": [
                '열기 : 후드를 열고 프론트 트렁크 열림 레버를 위로 누른 상태로 프론트 트렁크 커버를 들어 올리세요\n닫기 : 프론트 트렁크 커버를 아래로 눌러 듣으세요\n\n* 프론트 트렁크 커버 위를 누르거나 물건을 올려놓으면 변형 및 파손될 수 있습니다.\n* 프론트 트렁크 커버를 닫을 때 내부에 적재된 물건과 접촉하지 않도록 주의하세요. '
                ],
            "image": [
                './image/전면 트렁크.png'
                ],
            "type": 1,
        }
    },
    {
        "question": "주유/충전", 
        "index": 10,
        "response": {
            "text": [
                "충전도어",
                "충전방법",
                "충전량 확인하기"
                ],
            "image": [],
            "type": 0,
        }
    },
    {
        "question": "충전도어", 
        "index": 11,
        "response": {
            "text": [
                '아래 방법 중 하나를 선택하여 충전도어를 열고 닫을 수 있습니다.\n1. 충전도어 직접 터치\n2. 스마트키의 충전도어 열림/닫힘 버튼을 1초 이상 길게 누름\n\n* 주행 전 충전 도어가 잘 닫혔는지 확인하세요'
                ],
            "image": [
                './image/충전도어.png'
                ],
            "type": 1,
        }
    },
    {
        "question": "충전방법", 
        "index": 12,
        "response": {
            "text": [
                '충전도어를 열고, 어댑터의 커넥터부분을 찰칵 소리가 날 때까지 충분히 밀어 차량에 먼저 연결한 후, 어댑터의 인렛부부늘 충전기의 커넥터에 연결합니다.\n\n* 충전 커넥터와 충전 인렛에 먼지 등 이물질이 있는지 확인하십시오. \n* 커넥터를 정상적으로 결합하지 않을 경우 화재가 발생할 수 있습니다)\n\n충전이 완료되면 충전 커넷터 손잡이 부분을 잡고 잠금해제 버튼을 누르면서 당겨 분리하십시오\n\n*충전 후 인렛 커버가 제대로 닫히지 않은 상태로 충전 도어를 닫으면 관련 부품이 손상될 수 있습니다. '
                ],
            "image": [
                './image/충전방법.png'
                ],
            "type": 1,
        }
    },
    {
        "question": "충전량 확인하기", 
        "index": 13,
        "response": {
            "text": [
                    '구동용(고전압) 배터리 충전시 충전오어에서 충전량을 확인할 수 있습니다.\n\n*충전이 시작되면 계기판에 충전예상소요시간이 약 1분 동안 표시됩니다. '
                ],
            "image": [
                './image/충전량 확인하기.png'
                ],
            "type": 1,
        }
    },
    {
        "question": "경고등", 
        "index": 14,
        "response": {
            "text": [
                "주요 경고등 안내"
                ],
            "image": [],
            "type": 0,
        }
    },
    {
        "question": "주요 경고등 안내", 
        "index": 15,
        "response": {
            "text": [
                    '',
                ],
            "image": [
                './image/주요 경고등 안내.png'
                ],
            "type": 1,
        }
    },
    {
        "question": "직접입력", 
        "index": 999,
        "response": {
            "text": [],
            "image": [],
            "type": 2,
        }
    },

]

for i in sample_questions:
    response = i["response"]["text"]
    image = i["response"]["image"]
    button_type = i["response"]["type"]
    question = i["question"]

    response = json.dumps(response)
    image = json.dumps(image)

    cur.execute('''INSERT INTO button (question, response, image, type) VALUES (?, ?, ?, ?)''',
                (question, response, image, button_type))
    conn.commit()
conn.close()