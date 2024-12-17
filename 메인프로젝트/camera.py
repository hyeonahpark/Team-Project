import requests
import time
import os
import cv2
import pyttsx3

# 센서 데이터 URL
url = "http://192.168.0.98:8080/sensors.json"

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 180)
    engine.say(text)
    engine.runAndWait()

# 실시간 데이터 처리 함수
def monitor_sensor_data():
    try:
        while True:
            # 데이터 요청
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()

                # Gz 데이터 추출
                gravity_data = data.get('gravity', {}).get('data', [])
                if gravity_data:
                    # 마지막 데이터 가져오기
                    latest_timestamp, latest_values = gravity_data[-1]
                    gz_value = latest_values[2]  # Gz 값 (세 번째 값)

                    # 콘솔 내용 지우기
                    os.system('cls' if os.name == 'nt' else 'clear')

                    # 현재 값 출력
                    print(f"Timestamp: {latest_timestamp}, Gz: {gz_value}")

                    # Gz 값이 2.5 이하인 경우
                    if gz_value <= 2.5:
                        return '확인되었습니다.'
                    else:
                        return '카메라 각도를 조절해주세요.'
            else:
                print(f"Failed to fetch data: {response.status_code}")

            # 1초 간격으로 요청
            time.sleep(1)

    except Exception as e:
        print(f"An error occurred: {e}")

def camera_info(st, video_path):
    st.markdown("<h1 style='text-align: center;'>🏋️‍♂️ AI 홈트레이너</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>카메라 위치 조정</h3>", unsafe_allow_html=True)

    frame_placeholder = st.empty()
    text_placeholder = st.empty()
    count_placeholder = st.empty()

    ip_webcam_url = "http://192.168.0.98:8080/video"
    cap = cv2.VideoCapture(ip_webcam_url)  # 폰 카메라
    cap.set(cv2.CAP_PROP_FPS, 15)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("카메라에서 프레임을 읽을 수 없습니다.")
            break
        
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # BGR에서 RGB로 변환 (Streamlit은 RGB 형식을 사용)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Streamlit에 이미지 표시
        frame_placeholder.image(frame, channels="RGB", use_container_width=True)

        # 센서 데이터 확인
        cammera_info = monitor_sensor_data()
        if cammera_info == '확인되었습니다.':
            text_placeholder.markdown('카메라 설정이 확인되었습니다. 5초 뒤 운동을 시작합니다.', unsafe_allow_html=True)
            # speak('카메라 설정이 확인되었습니다. 10초 뒤 운동을 시작합니다.')
            cap.release()
            frame_placeholder.empty()
            text_placeholder.empty()
            
            countdown_placeholder = st.empty()
            video_placeholder = st.empty()
            
            # 영상 준비
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("비디오 파일을 열 수 없습니다.")
                return

            # 타이머 설정 (5초 카운트다운)
            countdown_start = 5
            start_time = time.time()

            while True:
                # 현재 시간 계산
                elapsed_time = time.time() - start_time

                # 타이머 UI 업데이트
                remaining_time = countdown_start - int(elapsed_time)
                if remaining_time > 0:
                    countdown_placeholder.markdown(
                        f"""
                        <div style="text-align: center; margin-top: 20px;">
                            <h2>운동 시작까지 {remaining_time}초...</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    countdown_placeholder.empty()  # 타이머 완료 후 제거

                # 영상 프레임 읽기
                ret, frame = cap.read()
                if ret:
                    # BGR에서 RGB로 변환 (Streamlit은 RGB 사용)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame, channels="RGB", use_container_width=True)
                else:
                    break  # 영상 끝

                # 타이머 종료 시 루프 종료
                if remaining_time <= 0 and not ret:
                    break

            # 자원 정리
            cap.release()
            video_placeholder.empty()
            text_placeholder.empty()
            st.session_state.current_exercise_state = "exercise"
            st.rerun()

        # 위치 조정 메시지
        text_placeholder.markdown('카메라 위치를 조정해주세요.', unsafe_allow_html=True)

    # 자원 정리
    st.rerun()
