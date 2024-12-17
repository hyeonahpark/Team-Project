# main.py
import streamlit as st
from pushup import run_pushup_session
from pushup import pushup_info
from plank import run_plank_session
from plank import plank_info
from crunch import run_crunch_session
from crunch import crunch_info
from lunge import run_lunge_session
from lunge import lunge_info
from squat import run_squat_session
from squat import squat_info
from camera import camera_info

import openai
import re
from ultralytics import YOLO
import time

# 운동별 예시 동영상 링크 
pushup_video = "C:/ai5/본프로젝트/메인/영상/샘플영상/푸시업.mp4"
plank_video = "C:/ai5/본프로젝트/메인/영상/샘플영상/플랭크.mp4"
crunch_video = "C:/ai5/본프로젝트/메인/영상/샘플영상/크런치.mp4"
lunge_video = "C:/ai5/본프로젝트/메인/영상/샘플영상/런지.mp4"
squat_video = "C:/ai5/본프로젝트/메인/영상/샘플영상/스쿼트.mp4"

# Streamlit 페이지 설정
st.set_page_config(page_title="AI 홈트레이너", layout="centered")

# 초기 상태 설정
if "stage" not in st.session_state:
    st.session_state.stage = "collecting_user_info"  # 단계: 정보 입력 단계

if "user_info" not in st.session_state:
    st.session_state.user_info = {
        "name": None,
        "age": None,
        "gender": None,
        "height": None,
        "weight": None,
        "difficulty": None,
        "exercise_types": [],
        "counts_or_time": None
    }

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": """당신은 홈트레이닝 ai입니다. 사용자의 정보를 받고 운동 횟수를 추천해줍니다. 
        다음 정보에 기반하여 사용자 질문에 존댓말로 간결하고 친절하게 답변해야 합니다. 
            :

        대화 순서 : 사용자 정보 입력받기 -> 운동 강도 선택 -> 운동 종류 제시 
        -> 사용자가 말한 운동 강도와 운동 종류에 따라 운동별 횟수 또는 운동시간 및 SET 측정 -> 사용자에게 확인 받기

        입력 받는 사용자 정보 : 이름, 나이, 성별, 키, 몸무게, 운동 경험, 운동 목표, 부상 여부
        운동 강도 : 초급, 중급, 고급 
        운동 종류 : 푸시업, 런지, 플랭크, 크런치, 스쿼트 
        운동 강도 및 사용자가 선택한 운동 종류에 따라 적절한 정확한 횟수 또는 정확한 운동 시간과 SET를 설정.
        푸시업, 런지, 크런치, 런지 : 횟수 및 세트 지정 (예: "런지: 10회 3세트")
        플랭크 : 초 및 세트 지정 (예: "플랭크: 70초 3세트")

        사용자에게 운동 추천을 제공할 때는 반드시 위의 형식으로 출력해주세요.
        운동 경험과 목표, 부상여부를 고려해서 추천을 제공해주세요. 
        """}]

# OpenAI API 호출 함수
def get_chatbot_response(user_input):
    st.session_state.messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=st.session_state.messages,
        max_tokens=1000,
        temperature=0.1
    )
    bot_response = response['choices'][0]['message']['content'].strip()
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    return bot_response

# 사용자 정보 입력 단계
if st.session_state.stage == "collecting_user_info":
    st.markdown("<h1 style='text-align: center;'>🏋️‍♂️ AI 홈트레이너</h1>", unsafe_allow_html=True)
    st.write("<p style='text-align: center;'>안녕하세요! 맞춤형 운동 계획을 위해 정보를 입력해주세요.</p>", unsafe_allow_html=True)

    user_info = st.session_state.user_info

    user_info["name"] = st.text_input("성함을 입력해주세요:", user_info["name"])
    user_info["age"] = st.number_input("나이를 입력해주세요:", min_value=0, max_value=100, value=user_info["age"] or 0, step=1)
    user_info["gender"] = st.selectbox("성별을 선택해주세요:", options=["남", "여"], index=0 if not user_info["gender"] else ["남", "여"].index(user_info["gender"]))
    user_info["height"] = st.number_input("키를 입력해주세요 (cm):", min_value=0, value=user_info["height"] or 0, step=1)
    user_info["weight"] = st.number_input("몸무게를 입력해주세요 (kg):", min_value=0, value=user_info["weight"] or 0, step=1)
    user_info["difficulty"] = st.selectbox("운동 강도를 선택해주세요:", options=["초급", "중급", "고급"], index=0 if not user_info["difficulty"] else ["초급", "중급", "고급"].index(user_info["difficulty"]))
    user_info["exercise_types"] = st.multiselect(
        "운동 종류를 선택해주세요 (다중 선택 가능):",
        options=["푸시업", "런지", "플랭크", "크런치", "스쿼트"],
        default=user_info["exercise_types"] or []
    )

    if st.button("다음으로"):
        st.session_state.stage = "asking_user_experience"
        st.rerun()

elif st.session_state.stage == "asking_user_experience":
    st.markdown("<h1 style='text-align: center;'>🏋️‍♂️ AI 홈트레이너</h1>", unsafe_allow_html=True)
    st.write("<p style='text-align: center;'>운동 경험 및 목표 설정</p>", unsafe_allow_html=True)

    user_info = st.session_state.user_info

    # 사용자 운동 경험 입력
    user_info["experience"] = st.radio(
        "운동 경험이 있으신가요?",
        options=["전혀 없음", "초급 (간단한 운동 경험 있음)", "중급 (정기적으로 운동)", "고급 (강도 높은 운동 경험)"],
        index=0
    )

    # 사용자 운동 목표 입력
    user_info["goal"] = st.radio(
        "운동 목표를 선택해주세요:",
        options=["체중 감량", "근력 강화", "유지/체력 증진", "재활"],
        index=0
    )

    # 아픈 부위 입력
    user_info["pain"] = st.text_input("현재 아프거나 부상 중인 부위가 있나요? (예: 무릎, 허리, 없음)")

    if st.button("다음으로"):
        st.session_state.stage = "chatbot_interaction"
        st.rerun()

elif st.session_state.stage == "chatbot_interaction":
    st.markdown("<h1 style='text-align: center;'>🏋️‍♂️ AI 홈트레이너</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>이전에 입력한 정보를 기반으로 맞춤형 운동 계획을 추천드립니다.</p>", unsafe_allow_html=True)

    # 초기 추천 결과를 세션 상태에 저장
    if "initial_recommendation" not in st.session_state:
        # 사용자 정보 기반 추천 요청 생성
        user_info = st.session_state.user_info
        exercise_request = (
            f"이름:{user_info['name']}, 나이:{user_info['age']}, 성별:{user_info['gender']}, "
            f"키:{user_info['height']}, 몸무게:{user_info['weight']}, 운동 강도: {user_info['difficulty']}, "
            f"운동 경험:{user_info['experience']}, 운동 목표:{user_info['goal']}, 부상 여부: {user_info['pain']}, "
            f"운동 종류: {', '.join(user_info['exercise_types'])}. "
            "위 정보를 기반으로 운동 횟수 또는 시간을 추천해주세요."
        )

        # OpenAI API 호출 및 운동 추천
        bot_response = get_chatbot_response(exercise_request)

        # 불필요한 내용 제거
        clean_response = re.sub(r"(당신은 홈트레이닝 ai입니다.*?출력\n)", "", bot_response, flags=re.DOTALL).strip()

        # 결과를 세션 상태에 저장
        st.session_state.initial_recommendation = clean_response

        # 추천 결과에서 운동 정보 추출
        st.session_state.user_info["counts_or_time"] = {
            '푸시업': re.search(r'푸시업:\s*(\d+)\s*회\s*(\d+)\s*세트', clean_response),
            '런지': re.search(r'런지:\s*(\d+)\s*회\s*(\d+)\s*세트', clean_response),
            '플랭크': re.search(r'플랭크:\s*(\d+)\s*초\s*(\d+)\s*세트', clean_response),
            '크런치': re.search(r'크런치:\s*(\d+)\s*회\s*(\d+)\s*세트', clean_response),
            '스쿼트': re.search(r'스쿼트:\s*(\d+)\s*회\s*(\d+)\s*세트', clean_response)
        }

        # 추출된 운동 정보를 세션 상태에 저장
        for exercise, match in st.session_state.user_info["counts_or_time"].items():
            if match:
                if exercise == '플랭크':
                    st.session_state.user_info["counts_or_time"][exercise] = {
                        "초": int(match.group(1)),
                        "세트": int(match.group(2))
                    }
                else:
                    st.session_state.user_info["counts_or_time"][exercise] = {
                        "횟수": int(match.group(1)),
                        "세트": int(match.group(2))
                    }
            else:
                st.session_state.user_info["counts_or_time"][exercise] = {"횟수" if exercise != '플랭크' else "초": 0, "세트": 0}

    # 챗봇 추천 내용 출력
    st.subheader("AI 추천 운동 계획")
    st.write(st.session_state.initial_recommendation)

    # 선택한 운동만 UI로 표시
    st.subheader("운동 조절하기")
    counts_or_time = st.session_state.user_info["counts_or_time"]
    selected_exercises = st.session_state.user_info["exercise_types"]  # 선택한 운동 리스트

    # 운동별 조절 UI
    for exercise in selected_exercises:
        if exercise in counts_or_time:
            values = counts_or_time[exercise]
            st.write(f"**{exercise}**")
            if exercise == '플랭크':
                default_seconds = values.get("초", 30)  # 기본값
                default_sets = values.get("세트", 1)   # 기본값

                # 초와 세트 선택
                seconds_index = max(0, min((default_seconds // 10) - 1, 5))  # 초 범위: [10, 20, 30, 40, 50, 60]
                sets_index = max(0, min(default_sets - 1, 4))                # 세트 범위: [1, 2, 3, 4, 5]

                counts_or_time[exercise]["초"] = int(st.selectbox(f"{exercise} 초", [10, 20, 30, 40, 50, 60], index=seconds_index))
                counts_or_time[exercise]["세트"] = int(st.selectbox(f"{exercise} 세트", [1, 2, 3, 4, 5], index=sets_index))
            else:
                default_reps = values.get("횟수", 10)  # 기본값
                default_sets = values.get("세트", 1)   # 기본값

                # 횟수와 세트 선택
                reps_index = max(0, min((default_reps - 1), 19))  # 횟수 범위: [1, 2, ..., 20]
                sets_index = max(0, min(default_sets - 1, 4))     # 세트 범위: [1, 2, 3, 4, 5]

                counts_or_time[exercise]["횟수"] = int(st.selectbox(f"{exercise} 횟수", list(range(1, 21)), index=reps_index))
                counts_or_time[exercise]["세트"] = int(st.selectbox(f"{exercise} 세트", [1, 2, 3, 4, 5], index=sets_index))

    # 수정된 운동 계획 표시
    st.write("### 수정된 운동 계획:")
    for exercise in selected_exercises:
        if exercise in counts_or_time:
            values = counts_or_time[exercise]
            if exercise == '플랭크':
                st.write(f"- {exercise}: {values['초']}초 {values['세트']}세트")
            else:
                st.write(f"- {exercise}: {values['횟수']}회 {values['세트']}세트")
            
    if st.button("다음으로"):
        st.session_state.stage = "exercise_session"
        st.rerun()


elif st.session_state.stage == "exercise_session":
    # 운동 시작 시간 기록
    if "start_time" not in st.session_state:
        st.session_state.start_time = time.time()

    # 현재 진행 중인 운동 추적
    if "current_exercise_index" not in st.session_state:
        st.session_state.current_exercise_index = 0  # 첫 번째 운동부터 시작

    exercises = st.session_state.user_info["exercise_types"]

    # 현재 운동 처리
    if st.session_state.current_exercise_index < len(exercises):
        current_exercise = exercises[st.session_state.current_exercise_index]

        # 운동별 상태 처리
        if "current_exercise_state" not in st.session_state:
            st.session_state.current_exercise_state = "camera_info"  # 초기 상태 설정

        if current_exercise == "푸시업":
            if st.session_state.current_exercise_state == "camera_info":
                pushup_info(st)
                
            elif st.session_state.current_exercise_state == "camera":
                camera_info(st, pushup_video)
                # camera_info(st)

            elif st.session_state.current_exercise_state == "exercise":
                feedback_messages = run_pushup_session(st, YOLO, st.session_state.user_info["counts_or_time"]['푸시업'])
                st.session_state.total_feedback = st.session_state.get("total_feedback", {})
                st.session_state.total_feedback["푸시업"] = feedback_messages
                st.session_state.current_exercise_index += 1
                st.session_state.current_exercise_state = 'camera_info'  # 다음 운동으로 초기화
                st.rerun()

        elif current_exercise == "런지":
            if st.session_state.current_exercise_state == "camera_info":
                lunge_info(st)
            
            elif st.session_state.current_exercise_state == "camera":
                camera_info(st, lunge_video)
                # camera_info(st)

            elif st.session_state.current_exercise_state == "exercise":
                # 런지 실행
                feedback_messages = run_lunge_session(st, YOLO, st.session_state.user_info["counts_or_time"]['런지'])
                st.session_state.total_feedback = st.session_state.get("total_feedback", {})
                st.session_state.total_feedback["런지"] = feedback_messages
                st.session_state.current_exercise_index += 1
                st.session_state.current_exercise_state = 'camera_info'
                st.rerun()

        elif current_exercise == "플랭크":
            if st.session_state.current_exercise_state == "camera_info":
                plank_info(st)
            
            elif st.session_state.current_exercise_state == "camera":
                camera_info(st, plank_video)
                # camera_info(st)

            elif st.session_state.current_exercise_state == "exercise":
                feedback_messages = run_plank_session(st, YOLO, st.session_state.user_info["counts_or_time"]['플랭크'])
                st.session_state.total_feedback = st.session_state.get("total_feedback", {})
                st.session_state.total_feedback["플랭크"] = feedback_messages
                st.session_state.current_exercise_index += 1
                st.session_state.current_exercise_state = 'camera_info'
                st.rerun()
        
        elif current_exercise == "크런치":
            if st.session_state.current_exercise_state == "camera_info":
                crunch_info(st)
            
            elif st.session_state.current_exercise_state == "camera":
                camera_info(st, crunch_video)
                # camera_info(st)

            elif st.session_state.current_exercise_state == "exercise":
                feedback_messages = run_crunch_session(st, YOLO, st.session_state.user_info["counts_or_time"]['크런치'])
                st.session_state.total_feedback = st.session_state.get("total_feedback", {})
                st.session_state.total_feedback["크런치"] = feedback_messages
                st.session_state.current_exercise_index += 1
                st.session_state.current_exercise_state = 'camera_info'
                st.rerun()

        elif current_exercise == "스쿼트":
            if st.session_state.current_exercise_state == "camera_info":
                squat_info(st)
            
            elif st.session_state.current_exercise_state == "camera":
                camera_info(st, squat_video)
                # camera_info(st)

            elif st.session_state.current_exercise_state == "exercise":
                feedback_messages = run_squat_session(st, YOLO, st.session_state.user_info["counts_or_time"]['스쿼트'])
                st.session_state.total_feedback = st.session_state.get("total_feedback", {})
                st.session_state.total_feedback["스쿼트"] = feedback_messages
                st.session_state.current_exercise_index += 1
                st.session_state.current_exercise_state = 'camera_info'
                st.rerun()

    else:
        # 모든 운동이 끝났을 때
        st.session_state.end_time = time.time()

        # 운동 시간 계산
        elapsed_time = int(st.session_state.end_time - st.session_state.start_time)
        hours = elapsed_time // 3600
        minutes = (elapsed_time % 3600) // 60
        seconds = elapsed_time % 60

        if hours > 0:
            st.session_state.exercise_time = f"{hours:02d}시 {minutes:02d}분 {seconds:02d}초"
        else:
            st.session_state.exercise_time = f"{minutes:02d}분 {seconds:02d}초"

        st.session_state.stage = "feedback_chatbot"
        st.rerun()

elif st.session_state.stage == 'feedback_chatbot':
    st.markdown("<h1 style='text-align: center;'>🏋️‍♂️ AI 홈트레이너</h1>", unsafe_allow_html=True)

    # 피드백 생성 및 저장
    if "feedback" not in st.session_state:
        # 사용자 정보 기반 피드백 요청 생성
        user_info = st.session_state.user_info
        total_feedback = st.session_state.get("total_feedback", {})
        exercise_feedback_request = (
            f"이름:{user_info['name']}, 나이:{user_info['age']}, 성별:{user_info['gender']}, "
            f"키:{user_info['height']}, 몸무게:{user_info['weight']}, "
            f"총 운동 시간:{st.session_state.exercise_time}, 피드백:{total_feedback}. "
            "위 정보를 기반으로 총 운동 시간, 운동별 소모 칼로리, 운동별 피드백을 출력 해주세요."
            "운동별 소모 칼로리는 나이, 성별, 키, 몸무게를 기준으로 계산하고, "
            "운동별 피드백은 피드백 내용을 기준으로 자연스럽게 조언해주세요."
        )

        # OpenAI API 호출 및 피드백 생성
        feedback_response = get_chatbot_response(exercise_feedback_request)

        # 피드백 결과 저장
        st.session_state.feedback = feedback_response

    # 최신 피드백 출력
    st.chat_message("assistant").markdown(f"**운동 피드백 결과:**\n\n{st.session_state.feedback}")

    # 사용자 입력 처리
    if user_input := st.chat_input("질문을 입력하세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": user_input})

        # OpenAI API 호출 및 응답 처리
        response = get_chatbot_response(user_input)

        # 응답 메시지 추가
        st.session_state.messages.append({"role": "assistant", "content": response})

        # 채팅창에 최신 대화 내용 표시
        st.chat_message("user").markdown(user_input)
        st.chat_message("assistant").markdown(response)

