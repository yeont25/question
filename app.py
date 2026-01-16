import streamlit as st
import google.generativeai as genai
import json
import re

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
            "temperature": 0.3,
            "top_p": 0.95,
            "max_output_tokens": 8192,
        }
    )
    return model

def parse_json_response(response_text):
    text = response_text.strip()
    
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            pass
    
    code_match = re.search(r'```\s*([\s\S]*?)\s*```', text)
    if code_match:
        try:
            return json.loads(code_match.group(1))
        except:
            pass
    
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass
    
    try:
        return json.loads(text)
    except:
        pass
    
    return None

CLASSIFICATION_PROMPT = '''당신은 교육 전문가입니다. 학생의 질문을 분석하세요.

[교육과정]
{curriculum}

[학생 질문]
{question}

[지시사항]
아래 JSON 형식으로만 응답하세요. 다른 텍스트 없이 JSON만 출력하세요.

```json
{{
    "relevance": {{
        "is_relevant": true,
        "reason": "이 질문이 교육과정과 관련된 이유를 설명"
    }},
    "bloom_taxonomy": {{
        "level": "기억",
        "explanation": "블룸 분류 수준 선택 이유"
    }},
    "question_quality": {{
        "score": 3,
        "feedback": "질문에 대한 피드백"
    }}
}}
```*
