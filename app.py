import streamlit as st
import google.generativeai as genai
import json

st.set_page_config(
    page_title="질문 분류 도우미",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .bloom-tag {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .relevant { background-color: #10B981; color: white; }
    .irrelevant { background-color: #EF4444; color: white; }
    .bloom-remember { background-color: #F59E0B; }
    .bloom-understand { background-color: #3B82F6; }
    .bloom-apply { background-color: #8B5CF6; }
    .bloom-analyze { background-color: #EC4899; }
    .bloom-evaluate { background-color: #14B8A6; }
    .bloom-create { background-color: #F97316; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_gemini():
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config={
            "temperature": 0.7,
            "top_p": 0.95,
            "max_output_tokens": 8192,
        }
    )
    return model

CLASSIFICATION_PROMPT = '''당신은 교육 전문가입니다. 학생의 질문을 분석하여 다음을 수행하세요.

## 교육과정 정보
{curriculum}

## 학생 질문
"{question}"

## 분석 요청
다음 JSON 형식으로 정확히 응답하세요:

{
    "relevance": {
        "is_relevant": true,
        "reason": "관련성 판단 이유"
    },
    "bloom_taxonomy": {
        "level": "기억/이해/적용/분석/평가/창조 중 하나",
        "explanation": "해당 수준으로 분류한 이유"
    },
    "question_quality": {
        "score": 3,
        "feedback": "질문의 질에 대한 피드백"
    }
}

JSON만 출력하고 다른 텍스트는 포함하지 마세요.'''

LEARNING_PATH_PROMPT = '''당신은 교육과정 전문가입니다. 학생의 질문을 해결하기 위한 학습 경로를 제시하세요.

## 교육과정 정보
{curriculum}

## 학생 질문
"{question}"

## 요청사항
이 질문을 해결하기 위해 학생이 배워야 할 내용을 교육과정을 바탕으로 체계적으로 정리해주세요.

다음 형식으로 응답하세요:

### 📚 필수 선수 지식
(이 질문을 이해하기 위해 먼저 알아야 할 개념들)

### 🎯 핵심 학습 내용
(질문과 직접 관련된 교육과정 내용)

### 🔗 연계 학습 주제
(질문을 확장하여 더 깊이 배울 수 있는 내용)

### 💡 추천 학습 활동
(이 질문을 탐구하기 위한 구체적인 활동 제안)

### 📖 참고할 교육과정 성취기준
(관련된 성취기준 나열)'''

if "messages" not in st.session_state:
    st.session_state.messages = []
if "questions_history" not in st.session_state:
    st.session_state.questions_history = []
if "curriculum" not in st.session_state:
    st.session_state.curriculum = ""

with st.sidebar:
    st.markdown("## 📋 교육과정 설정")
    
    curriculum_input = st.text_area(
        "교육과정 내용을 입력하세요",
        value=st.session_state.curriculum,
        height=300,
        placeholder="예시:\n[과목] 초등학교 5학년 과학\n[단원] 태양계와 별\n[성취기준]\n- 태양이 지구의 에너지원임을 이해한다."
    )
    
    if st.button("✅ 교육과정 저장", use_container_width=True):
        st.session_state.curriculum = curriculum_input
        st.success("교육과정이 저장되었습니다!")
    
    st.divider()
    
    st.markdown("## 🎓 Bloom's Taxonomy")
    st.markdown("""
| 수준 | 설명 |
|------|------|
| 🟡 **기억** | 사실, 용어 회상 |
| 🔵 **이해** | 의미 파악, 설명 |
| 🟣 **적용** | 새로운 상황에 적용 |
| 🩷 **분석** | 구성요소 분해 |
| 🩵 **평가** | 판단, 비평 |
| 🟠 **창조** | 새로운 것 생성 |
    """)
    
    st.divider()
    
    if st.button("🗑️ 대화 기록 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.questions_history = []
        st.rerun()

st.markdown('<h1 class="main-header">🎓 질문 분류 도우미</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">학생 질문을 분석하고 학습 경로를 제시합니다</p>', unsafe_allow_html=True)

if not st.session_state.curriculum:
    st.warning("⚠️ 먼저 사이드바에서 교육과정을 입력해주세요.")

tab1, tab2 = st.tabs(["💬 질문 분석", "📊 질문 기록"])

with tab1:
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                with st.chat_message("user", avatar="👨‍🎓"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(msg["content"], unsafe_allow_html=True)
    
    user_question = st.chat_input("학생 질문을 입력하세요...")
    
    if user_question and st.session_state.curriculum:
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        with st.chat_message("user", avatar="👨‍🎓"):
            st.markdown(user_question)
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("질문을 분석하고 있습니다..."):
                try:
                    model = init_gemini()
                    
                    classification_prompt = CLASSIFICATION_PROMPT.format(
                        curriculum=st.session_state.curriculum,
                        question=user_question
                    )
                    
                    response = model.generate_content(classification_prompt)
                    response_text = response.text.strip()
                    
                    if "```json" in response_text:
                        json_str = response_text.split("```json")[1].split("```")[0]
                    elif "```" in response_text:
                        json_str = response_text.split("```")[1].split("```")[0]
                    else:
                        json_str = response_text
                    
                    analysis = json.loads(json_str.strip())
                    
                    relevance = analysis.get("relevance", {})
                    bloom = analysis.get("bloom_taxonomy", {})
                    quality = analysis.get("question_quality", {})
                    
                    is_relevant = relevance.get("is_relevant", False)
                    relevance_class = "relevant" if is_relevant else "irrelevant"
                    relevance_text = "✅ 수업 관련" if is_relevant else "❌ 수업 무관"
                    
                    bloom_level = bloom.get("level", "미분류")
                    bloom_colors = {
                        "기억": "bloom-remember",
                        "이해": "bloom-understand", 
                        "적용": "bloom-apply",
                        "분석": "bloom-analyze",
                        "평가": "bloom-evaluate",
                        "창조": "bloom-create"
                    }
                    bloom_class = bloom_colors.get(bloom_level, "bloom-remember")
                    
                    score = quality.get("score", 3)
                    stars = "⭐" * score
                    
                    result_html = f'''### 📊 질문 분석 결과

<span class="bloom-tag {relevance_class}">{relevance_text}</span>
<span class="bloom-tag {bloom_class}">Bloom: {bloom_level}</span>
<span class="bloom-tag" style="background-color: #6366F1; color: white;">품질: {stars}</span>

**📌 관련성 분석**
> {relevance.get("reason", "분석 중...")}

**🎯 Bloom taxonomy 분류**
> {bloom.get("explanation", "분석 중...")}

**💬 질문 피드백**
> {quality.get("feedback", "분석 중...")}'''
                    
                    st.markdown(result_html, unsafe_allow_html=True)
                    
                    if is_relevant:
                        if st.button("📚 학습 경로 보기", key=f"path_{len(st.session_state.messages)}"):
                            with st.spinner("학습 경로를 생성하고 있습니다..."):
                                learning_prompt = LEARNING_PATH_PROMPT.format(
                                    curriculum=st.session_state.curriculum,
                                    question=user_question
                                )
                                learning_response = model.generate_content(learning_prompt)
                                st.markdown("---")
                                st.markdown(learning_response.text)
                    
                    st.session_state.questions_history.append({
                        "question": user_question,
                        "is_relevant": is_relevant,
                        "bloom_level": bloom_level,
                        "score": score,
                        "analysis": analysis
                    })
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result_html
                    })
                    
                except json.JSONDecodeError as e:
                    st.error(f"응답 파싱 오류: {e}")
                    st.code(response_text)
                except Exception as e:
                    st.error(f"오류 발생: {e}")

with tab2:
    if st.session_state.questions_history:
        st.markdown("### 📋 분석된 질문 목록")
        
        col1, col2 = st.columns(2)
        with col1:
            filter_relevance = st.selectbox(
                "관련성 필터",
                ["전체", "수업 관련", "수업 무관"]
            )
        with col2:
            filter_bloom = st.selectbox(
                "Bloom 수준 필터",
                ["전체", "기억", "이해", "적용", "분석", "평가", "창조"]
            )
        
        filtered = st.session_state.questions_history.copy()
        
        if filter_relevance == "수업 관련":
            filtered = [q for q in filtered if q["is_relevant"]]
        elif filter_relevance == "수업 무관":
            filtered = [q for q in filtered if not q["is_relevant"]]
            
        if filter_bloom != "전체":
            filtered = [q for q in filtered if q["bloom_level"] == filter_bloom]
        
        for i, q in enumerate(filtered):
            with st.expander(f"{'✅' if q['is_relevant'] else '❌'} {q['question'][:50]}..."):
                st.markdown(f"**질문:** {q['question']}")
                st.markdown(f"**Bloom 수준:** {q['bloom_level']}")
                st.markdown(f"**품질 점수:** {'⭐' * q['score']}")
                
                if q["is_relevant"]:
                    if st.button("📚 이 질문의 학습 경로 생성", key=f"history_{i}"):
                        with st.spinner("학습 경로를 생성하고 있습니다..."):
                            model = init_gemini()
                            learning_prompt = LEARNING_PATH_PROMPT.format(
                                curriculum=st.session_state.curriculum,
                                question=q["question"]
                            )
                            learning_response = model.generate_content(learning_prompt)
                            st.markdown("---")
                            st.markdown(learning_response.text)
        
        st.markdown("---")
        st.markdown("### 📈 질문 통계")
        
        total = len(st.session_state.questions_history)
        relevant = sum(1 for q in st.session_state.questions_history if q["is_relevant"])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("총 질문 수", total)
        col2.metric("수업 관련 질문", relevant)
        col3.metric("관련성 비율", f"{(relevant/total*100):.1f}%" if total > 0 else "0%")
        
        bloom_counts = {}
        for q in st.session_state.questions_history:
            level = q["bloom_level"]
            bloom_counts[level] = bloom_counts.get(level, 0) + 1
        
        if bloom_counts:
            st.markdown("#### Bloom's Taxonomy 분포")
            for level, count in sorted(bloom_counts.items()):
                st.progress(count/total, text=f"{level}: {count}개 ({count/total*100:.1f}%)")
    
    else:
        st.info("아직 분석된 질문이 없습니다. '질문 분석' 탭에서 질문을 입력해주세요.")

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #9CA3AF;'>🎓 교육과정 기반 질문 분류 시스템 | Powered by Gemini 2.5 Flash</p>",
    unsafe_allow_html=True
)
