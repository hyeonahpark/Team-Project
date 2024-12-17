# pushup.py
import cv2
import joblib
import math
import numpy as np
from PIL import ImageFont, Image, ImageDraw
import mediapipe as mp
import time
from ultralytics import YOLO

label = ['정자세', '오자세']
model = YOLO('yolov5/yolo11n.pt')
plank_model = joblib.load('C:/ai5/본프로젝트/메인/_data/운동별 모델/plank.pkl')
font_path = "C:/ai5/본프로젝트/메인/_data/강원교육튼튼.ttf"
font = ImageFont.truetype(font_path, 20)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def plank_info(st):
    st.markdown("<h1 style='text-align: center;'>🏋️‍♂️ AI 홈트레이너</h1>", unsafe_allow_html=True)

    with st.container():
        st.markdown("""
                <div style="
                    background-color: #fff; 
                    padding: 30px; 
                    border-radius: 15px; 
                    border: 1px solid #ddd; 
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); 
                    max-width: 600px; 
                    margin: auto;
                    font-family: 'Roboto', Arial, sans-serif;
                ">
                    <div>
                        <h2 style="font-size: 20px; color: #333; margin: 0; text-align: center;">플랭크</h3>
                        <h3 style="font-size: 18px; color: #333; margin: 0;">1. 카메라 설정 안내</h3>
                        <ul style="font-size: 14px; color: #555; line-height: 1.8; padding-left: 20px; margin-top: 10px;">
                            <li>휴대폰을 바닥에 세워두시고, <b>좌측</b> 또는 <b>우측 전신</b>이 화면에 잘 나오도록 카메라를 조정해주세요.</li>
                            <li>주변 환경이 밝고, 촬영에 방해되지 않도록 정리해주세요.</li>
                        </ul>
                    </div>
                    <div style="
                        border-top: 1px solid #eee; 
                        padding-top: 20px; 
                        margin-top: 20px;
                    ">
                        <h3 style="font-size: 18px; color: #333; margin: 0;">2. 운동 시작</h3>
                        <ul style="font-size: 14px; color: #555; line-height: 1.8; padding-left: 20px; margin-top: 10px;">
                            <li>준비가 완료되셨다면, '다음' 버튼을 눌러주세요!</li>
                            <li>운동이 바로 시작됩니다!</li>
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.image("C:/ai5/본프로젝트/메인/_data/플랭크.png", use_container_width=True)
    
        if st.button("다음"):
            st.session_state.current_exercise_state = "camera"
            st.rerun()

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)


def calculate_angle(point_a, point_b, point_c):
    ab = (point_b.x - point_a.x, point_b.y - point_a.y, point_b.z - point_a.z)
    bc = (point_c.x - point_b.x, point_c.y - point_b.y, point_c.z - point_b.z)
    dot_product = sum(a * b for a, b in zip(ab, bc))
    magnitude_ab = math.sqrt(sum(a**2 for a in ab))
    magnitude_bc = math.sqrt(sum(b**2 for b in bc)) 
    if magnitude_ab == 0 or magnitude_bc == 0:
        return 0.0
    return math.degrees(math.acos(dot_product / (magnitude_ab * magnitude_bc)))


def flip_landmarks(landmarks):
    flipped = [None] * len(landmarks)
    swap_indices = {
        0: 0, 1: 4, 2: 5, 3: 6, 4: 1, 5: 2, 6: 3, 7: 8, 8: 7,
        9: 10, 10: 9, 11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15,
        17: 18, 18: 17, 19: 20, 20: 19, 21: 22, 22: 21, 23: 24, 24: 23,
        25: 26, 26: 25, 27: 28, 28: 27, 29: 30, 30: 29, 31: 32, 32: 31,
    }
    for idx, lm in enumerate(landmarks):
        new_idx = swap_indices.get(idx, idx)
        flipped[new_idx] = type(lm)(
            x=1.0 - lm.x,
            y=lm.y,
            z=lm.z,
            visibility=lm.visibility
        )
    return flipped

##################### 플랭크로 변경 필요 #####################
def run_plank_session(st, YOLO, info):
    # 초기 변수 설정
    plank_time_correct = 0  # 정자세 누적 시간
    plank_time_wrong = 0  # 오자세 누적 시간
    total_history = []  # 전체 프레임의 예측 결과 저장
    correct_threshold = 0.5  # 정자세 비율 기준 (70%)
    current_state = "대기"  # 초기 상태는 "대기"
    
    # 세트 및 목표 시간 설정
    target_time = info["초"]  # 목표 시간 (초 단위)
    target_sets = info["세트"]  # 목표 세트
    completed_sets = 0  # 완료된 세트 수

    frame_skip = 4  # 1 프레임 건너뛰기
    frame_count = 0  # 프레임 카운터 초기화
    start_time = None  # 정자세 시작 시간
    is_holding = False  # 정자세 유지 여부

    # 결과 저장용 딕셔너리 초기화
    results_dict = {
        "정자세 시간": 0,
        "오자세 시간": 0,
        "어깨와 손의 위치를 일치하게 해주세요.": 0,
        "목과 허리, 엉덩이, 다리가 일직선이 되도록 해주세요.": 0,
        "엉덩이를 양쪽 균형 있게 유지하세요.": 0,
        "고개가 정면을 향하게 해주세요.": 0,
    }

    # MediaPipe Pose 초기화
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # 스트림릿 화면 초기화
    st.markdown("<h1 style='text-align: center;'>플랭크</h1>", unsafe_allow_html=True)

    # 스트리밍 비디오 피드 설정
    video_url = "C:/ai5/본프로젝트/메인/영상/플랭크/사영1.mp4"
    ip_webcam_url = "http://192.168.0.98:8080/video"    

    # 채팅 메시지 창
    frame_placeholder = st.empty()
    chat_placeholder = st.empty()
    time_placeholder = st.empty()
    set_placeholder = st.empty()  # 세트 표시용

    # 비디오 스트리밍 처리
    cap = cv2.VideoCapture(video_url) # 동영상
    # cap = cv2.VideoCapture(ip_webcam_url) # 폰 카메라
    # cap = cv2.VideoCapture(0)       # 웹캠
    
    cap.set(cv2.CAP_PROP_FPS, 15)    
    
    if "feedback_display_start_time" not in st.session_state:
        st.session_state.feedback_display_start_time = None 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:  # 프레임 건너뛰기
            continue
        
        # 프레임 회전
        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)


        # YOLO 모델로 사람 바운딩 박스 생성
        results = model(frame)
        best_box = None
        best_confidence = 0.0

        for result in results:
            for box in result.boxes:
                class_id = box.cls[0]
                confidence = box.conf[0]
                if int(class_id) == 0 and confidence > best_confidence:
                    best_box = box
                    best_confidence = confidence

        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            person_roi = frame[y1:y2, x1:x2]

            with mp_pose.Pose(static_image_mode=False, model_complexity=2) as pose:
                results_pose = pose.process(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))
                if results_pose.pose_landmarks:
                    try:
                        landmarks = results_pose.pose_landmarks.landmark

                        # 좌우 반전 감지
                        nose_x = landmarks[0].x
                        hip_mid_x = (landmarks[23].x + landmarks[24].x) / 2
                        is_right_facing = nose_x > hip_mid_x
                        
                        if is_right_facing:
                            landmarks = flip_landmarks(landmarks)
                        else:
                            landmarks = landmarks
                            
                        # 랜드마크 데이터 생성
                        landmarks_data = []
                        for idx in [11, 12, 23, 24, 25, 26, 27, 28]:
                            print(f"Landmark {idx}: X={landmarks[idx].x}, Y={landmarks[idx].y}, Z={landmarks[idx].z}")
                            landmarks_data.extend([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])

                        left_hip_angle = calculate_angle(landmarks[11], landmarks[23], landmarks[25])
                        right_hip_angle = calculate_angle(landmarks[12], landmarks[24], landmarks[26])
                        left_knee_angle = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
                        right_knee_angle = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
                        shoulder_balance = abs(landmarks[11].y - landmarks[12].y)
                        hip_balance = abs(landmarks[23].y - landmarks[24].y)

                        landmarks_data.extend([
                            left_hip_angle, right_hip_angle,
                            left_knee_angle, right_knee_angle,
                            shoulder_balance, hip_balance
                        ])

                        prediction = plank_model.predict([landmarks_data])
                        prediction_text = label[int(prediction[0])]

                        total_history.append(prediction_text == "정자세")
                        correct_ratio = sum(total_history) / len(total_history)
                        
                        # 시간 계산
                        if correct_ratio >= correct_threshold:
                            if not is_holding:
                                start_time = time.time()
                                is_holding = True
                            else:
                                elapsed_time = time.time() - start_time
                                plank_time_correct += elapsed_time
                                start_time = time.time()

                                # 세트 완료 처리
                                while plank_time_correct >= target_time:  # 초과된 시간도 계산에 반영
                                    completed_sets += 1
                                    total_history = []
                                    plank_time_correct -= target_time  # 초과된 시간을 다음 세트로 넘김

                                    if completed_sets >= target_sets:
                                        st.success("모든 세트를 완료했습니다! 수고하셨습니다!")
                                        cap.release()
                                        cv2.destroyAllWindows()
                                        return results_dict
                        else:
                            if is_holding:
                                elapsed_time = time.time() - start_time
                                plank_time_wrong += elapsed_time
                                is_holding = False
                                start_time = None

                        # "오자세"인 경우 피드백 생성 및 출력
                        if prediction_text == "오자세":
                            feedback_messages = []

                            # 어깨와 손 위치 불일치 피드백
                            shoulder_wrist_distance_left = calculate_distance(landmarks[11], landmarks[15])
                            shoulder_wrist_distance_right = calculate_distance(landmarks[12], landmarks[16])
                            if shoulder_wrist_distance_left > 0.15 or shoulder_wrist_distance_right > 0.15:
                                results_dict["어깨와 손의 위치를 일치하게 해주세요."] += 1
                                feedback_messages.append("어깨와 손의 위치를 일치하게 해주세요.")

                            # 목-허리-엉덩이-다리 일직선 피드백
                            body_alignment_angle = calculate_angle(landmarks[0], landmarks[23], landmarks[27])
                            if body_alignment_angle < 170:
                                results_dict["목과 허리, 엉덩이, 다리가 일직선이 되도록 해주세요."] += 1
                                feedback_messages.append("목과 허리, 엉덩이, 다리가 일직선이 되도록 해주세요.")

                            # 엉덩이 높이 불균형 피드백
                            hip_height_difference = abs(landmarks[23].y - landmarks[24].y)
                            if hip_height_difference > 0.05:
                                results_dict["엉덩이를 양쪽 균형 있게 유지하세요."] += 1
                                feedback_messages.append("엉덩이를 양쪽 균형 있게 유지하세요.")

                            # 고개 정면 피드백
                            nose_x = landmarks[0].x
                            shoulder_center_x = (landmarks[11].x + landmarks[12].x) / 2
                            if abs(nose_x - shoulder_center_x) > 0.1:
                                results_dict["고개가 정면을 향하게 해주세요."] += 1
                                feedback_messages.append("고개가 정면을 향하게 해주세요.")

                            # 피드백 출력
                            if feedback_messages:
                                feedback_text = "<br>".join(feedback_messages)
                                chat_placeholder.markdown(feedback_text, unsafe_allow_html=True)
                                st.session_state.feedback_display_start_time = time.time()
                        else:
                            # "정자세"일 경우 피드백 제거
                            if st.session_state.feedback_display_start_time and time.time() - st.session_state.feedback_display_start_time > 5:
                                chat_placeholder.markdown("", unsafe_allow_html=True)
                                st.session_state.feedback_display_start_time = None
                        
                        # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        # pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        # draw = ImageDraw.Draw(pil_img)
                        # draw.text(
                        #     (x1, y1 - 40),
                        #     f"Plank Time: {int(plank_time_correct)}s | Ratio: {correct_ratio:.2f} | {prediction_text}",
                        #     font=font,
                        #     fill=(255, 0, 0),
                        # )
                        # frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                        # mp_drawing.draw_landmarks(person_roi, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        
                        # 플랭크 시간 및 상태 출력
                        time_placeholder.info(f"정자세 시간: {int(plank_time_correct)}s /{int(target_time)}s    플랭크 오자세 시간 : {plank_time_wrong}s    세트: {completed_sets}/{target_sets}")
                        # time_placeholder.info(f"정자세 시간: {int(plank_time_correct)}s /{int(target_time)}s   세트: {completed_sets}/{target_sets}")

                    except IndexError:
                        print("IndexError: 랜드마크 데이터 추출 오류")

        

        # 회전된 프레임을 표시
        frame_placeholder.image(frame, channels="BGR")
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # 결과 요약
    results_dict["정자세 시간"] = plank_time_correct
    results_dict["오자세 시간"] = plank_time_wrong
    total_feedback = results_dict["어깨와 손의 위치를 일치하게 해주세요."] + results_dict["목과 허리, 엉덩이, 다리가 일직선이 되도록 해주세요."] + results_dict["엉덩이를 양쪽 균형 있게 유지하세요."] + results_dict["고개가 정면을 향하게 해주세요."]
    results_dict["어깨와 손의 위치를 일치하게 해주세요."] = round(results_dict["어깨와 손의 위치를 일치하게 해주세요."] / total_feedback, 5)
    results_dict["목과 허리, 엉덩이, 다리가 일직선이 되도록 해주세요."] = round(results_dict["목과 허리, 엉덩이, 다리가 일직선이 되도록 해주세요."] / total_feedback , 5)
    results_dict["엉덩이를 양쪽 균형 있게 유지하세요."] = round(results_dict["엉덩이를 양쪽 균형 있게 유지하세요."] / total_feedback, 5)
    results_dict["고개가 정면을 향하게 해주세요."] = round(results_dict["고개가 정면을 향하게 해주세요."] / total_feedback, 5)
        
    return results_dict
