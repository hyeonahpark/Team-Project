from flask import Flask, render_template, request, jsonify
import openai
import requests
import sounddevice as sd
from scipy.io.wavfile import read
import io
import random
from datetime import datetime

app = Flask(__name__)

# OpenAI와 Typecast API 키 설정
API_TOKEN = '__plt9onJuVApvTfRbQH4dufxRQAk8wmsKx7vt9eUpcME'
HEADERS = {'Authorization': f'Bearer {API_TOKEN}'}

# 초기 메시지 설정
messages = [
    {
        "role": "system",
        "content": """안녕하세요, 누리카세 예약 도우미입니다. 저는 누리카세 식당 예약을 도와드리는 챗봇입니다. 
        아래의 정보를 바탕으로 사용자의 질문에 친절하고 간결하게 답변합니다. 예약 관련 정보가 정확할 경우 추가 질문을 하지 않습니다.
        
        **식당 정보**
        - 가게 이름: 누리카세
        - 운영 시간: 매일 12:00 ~ 22:00
        - 예약 가능 시간: 오후 12시 ~ 오후 2시, 오후 5시 ~ 오후 9시
        - 브레이크타임: 15:00 ~ 17:00
        - 최대 좌석 수: 8석
        - 메뉴: 런치 오마카세, 디너 오마카세
        - 주차: 근처 공영주차장 이용
        - 예약인원 최대: 8명까지 가능
        - 당일 예약 불가, 연중무휴 운영
        - 예약 절차: 예약 날짜 -> 예약 시간 -> 인원 -> 예약자 성함 -> 연락처(휴대폰 뒷번호 4자리) -> 예약 정보 확인
        
        **예약 정보**
        - 접수번호 (랜덤 4자리 숫자)
        - 접수 날짜 및 시간 (현재 날짜와 시간 자동 생성)
        - 예약 날짜, 예약 시간, 인원수, 예약자 성함, 연락처 뒷번호

        **예약 완료 후 절차**
        1. 사용자에게 예약 정보를 확인
        2. 맞다고 하면 "예약이 완료되었습니다."라고 안내합니다.
        3. 그 후 알러지 여부확인. 알러지 여부는 예약 중에 한 번만 물어볼 것
        4. 모든 절차가 끝나면 추가로 필요한 도움이 있는지 물어보고, 추가 요청이 없으면 "통화가 종료됩니다."
        """
    }
]

def get_chatbot_response(user_input):
    global messages
    
    # 접수 날짜와 시간 설정
    reception_date = datetime.now().strftime("%m월 %d일")
    reception_time = datetime.now().strftime("%H시 %M분")
    reception_number = str(random.randint(1000, 9999))  # 무작위 접수 번호 생성

    # 접수 정보를 OpenAI API에 전달하기 위해 messages에 추가
    messages.append({"role": "user", "content": user_input})
    

    # OpenAI API 호출
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=300,  # 필요한 답변 길이에 맞춰 토큰 수 조정
        temperature=0.2
    )
    bot_response = response['choices'][0]['message']['content'].strip()
    messages.append({"role": "assistant", "content": bot_response})
    messages.append({
        "role": "system",
        "content": (
            f"예약 완료 절차에서 사용자에게 예약 정보를 확인할 때만 다음 정보를 포함하여 안내하세요: "
            f"예약 날짜, 예약 시간, 인원수, 예약자 성함, 휴대폰 뒷번호, 그리고 접수 정보 ("
            f"접수 날짜: {reception_date}, 접수 시간: {reception_time}, 접수 번호: {reception_number})"
        )
    })

    return bot_response

def speak(text):
    r = requests.post('https://typecast.ai/api/speak', headers=HEADERS, json={
        'text': text,
        'lang': 'auto',
        'actor_id': '661797923ed12f31b61c4b5f',
        'xapi_hd': True,
        'model_version': 'latest'
    })
    speak_url = r.json()['result']['speak_v2_url']
    
    for _ in range(120):
        r = requests.get(speak_url, headers=HEADERS)
        ret = r.json()['result']
        
        if ret['status'] == 'done':
            audio_data = requests.get(ret['audio_download_url']).content
            audio_stream = io.BytesIO(audio_data)
            sample_rate, audio = read(audio_stream)
            sd.play(audio, samplerate=sample_rate)
            sd.wait()
            break

@app.route('/')
def home():
    start_message = "최고의 고객께, 최고의 서비스를, 안녕하세요, 누리카세입니다. 무엇을 도와드릴까요?"
    return render_template('index.html', start_message=start_message)

@app.route('/speak', methods=['POST'])
def speak_text():
    text = request.json['text']
    speak(text)
    return jsonify({'status': 'completed'})

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json['user_input']
    bot_response = get_chatbot_response(user_input)
    speak(bot_response)
    return jsonify({'user_input': user_input, 'bot_response': bot_response})

if __name__ == "__main__":
    app.run(debug=True)
