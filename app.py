# 질문 입력
col1, col2 = st.columns([4, 1])

with col1:
    user_question = st.chat_input("학생 질문을 입력하세요...")

# 질문 처리
if user_question and st.session_state.curriculum:
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": user_question})
    
    with st.chat_message("user", avatar="👨‍🎓"):
        st.markdown(user_question)
    
    # AI 응답 생성
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("질문을 분석하고 있습니다..."):
            try:
                model = init_gemini()
                
                # 질문 분류
                classification_prompt = CLASSIFICATION_PROMPT.format(
                    curriculum=st.session_state.curriculum,
                    question=user_question
                )
                
                response = model.generate_content(classification_prompt)
                response_text = response.text.strip()
                
                # JSON 파싱
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].split("```")[0]
                else:
                    json_str = response_text
                
                analysis = json.loads(json_str.strip())
                
                # 결과 포맷팅
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
                
                result_html = f"""
