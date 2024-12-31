# 가상환경 : noori_test1
import streamlit as st
import openai
import speech_recognition as sr
from gtts import gTTS
import os
import playsound
import threading

def select_language_ui():
    languages = {
        "한국어": "ko",
        "영어": "en",
        "일본어": "ja",
        "중국어": "zh",
        "독일어": "de",
        "스페인어": "es"
    }
    input_lang_name = st.selectbox("입력 언어를 선택하세요:", options=list(languages.keys()))
    target_lang_name = st.selectbox("번역 언어를 선택하세요:", options=list(languages.keys()))

    return languages[input_lang_name], languages[target_lang_name], input_lang_name, target_lang_name

def transcribe_speech_to_text(input_lang_code):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            audio = recognizer.listen(source, timeout=10)
            text = recognizer.recognize_google(audio, language=input_lang_code)
            return text
        except sr.UnknownValueError:
            st.error("음성을 인식하지 못했습니다. 다시 시도해주세요.")
        except sr.RequestError as e:
            st.error(f"Google Speech Recognition 서비스에 문제가 발생했습니다: {e}")
        return None

def translate_text_with_openai(text, target_language="en"):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = f"Translate the following text to {target_language}:\n{text}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful translation assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        translation = response['choices'][0]['message']['content'].strip()
        return translation
    except Exception as e:
        st.error(f"번역 중 오류가 발생했습니다: {e}")
        return None

def text_to_speech(text, lang="en"):
    try:
        tts = gTTS(text=text, lang=lang)
        filename = "output.mp3"
        tts.save(filename)

        # 별도의 스레드에서 재생
        def play_audio():
            playsound.playsound(filename)
            os.remove(filename)
        
        thread = threading.Thread(target=play_audio)
        thread.start()
    except Exception as e:
        st.error(f"TTS 생성 중 오류가 발생했습니다: {e}")

def main():
    st.title("🌍 동시통역 프로그램")

    with st.sidebar:
        st.header("🌐 언어 설정")
        input_lang_code, target_lang_code, input_lang_name, target_lang_name = select_language_ui()

    st.markdown(f"""
        <style>
        .chat-container {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .chat-message {{
            border-radius: 10px;
            padding: 10px;
            max-width: 80%;
            margin-bottom: 10px;
        }}
        .user-message {{
            align-self: flex-end;
            background-color: #0078D7;
            color: #FFFFFF;
        }}
        .bot-message {{
            align-self: flex-start;
            background-color: #F1F0F0;
            color: #333;
        }}
        </style>
    """, unsafe_allow_html=True)
    chat_messages = st.empty()
    
    chat_messages = []  # 채팅 메시지 저장용 리스트
    
    if st.button("말하기"):
        input_text = transcribe_speech_to_text(input_lang_code)
        if input_text:
            chat_messages.append({"type": "user", "content": input_text})
            translated_text = translate_text_with_openai(input_text, target_language=target_lang_code)
            if translated_text:
                chat_messages.append({"type": "bot", "content": translated_text})
                text_to_speech(translated_text, lang=target_lang_code)

        # 채팅 메시지 렌더링
        for msg in chat_messages:
            if msg["type"] == "user":
                st.markdown(f"""
                    <div class="chat-container">
                        <div class="chat-message user-message">
                            {msg["content"]}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            elif msg["type"] == "bot":
                st.markdown(f"""
                    <div class="chat-container">
                        <div class="chat-message bot-message">
                            {msg["content"]}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
