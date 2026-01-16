import streamlit as st
import google.generativeai as genai
from typing import Optional
import json

# =====================
# 페이지 설정
# =====================
st.set_page_config(
    page_title="질문 분류 도우미",
    page_icon="🎓",
    layout="wide"
)

# =====================
# 스타일 설정
# =====================
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
    .question-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
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
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #E0E7FF;
        margin-left: 2rem;
    }
    .assistant-message {
        background-color: #F3F4F6;
        margin-right: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# =====================
# Gemini API 설정
# =====================
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

# =====================
# 프롬프트 템플릿
# =====================
CLASSIFICATION_PROMPT = """
당신은 교육 전문가입니다. 학생의 질문을 분석하여 다음을 수행하세요.

## 교육과정 정보
{curriculum}

## 학생 질문
"{question}"

## 분석 요청
다음 JSON 형식으로 정확히 응답하세요:

```json
{{
    "relevance": {{
        "is_relevant": true/false,
        "reason": "관련성 판단 이유 (2-3문장)"
    }},
    "bloom_taxonomy": {{
        "level": "기억/이해/적용/분석/평가/창조 중 하나",
        "explanation": "해당 수준으로 분류한 이유"
    }},
    "question_quality": {{
        "score": 1-5,
        "feedback": "질문의 질에 대한 피드백"
    }}
}}
