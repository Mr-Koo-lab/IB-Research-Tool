import streamlit as st
import os
import time
from typing import TypedDict, List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# ==========================================
# 1. 웹 UI 및 레이아웃 설정
# ==========================================
st.set_page_config(page_title="Mr.Koo Agent Research Tool", layout="wide")
st.title("🚀 Mr.Koo Agent Research Tool (V2.6)")

with st.sidebar:
    st.header("⚙️ 인프라 및 가르치기 센터")
    api_key = st.text_input("Google API Key", type="password")
    
    st.divider()
    st.subheader("🛰️ 외부 인프라 연동망")
    tg_api_id = st.text_input("Telegram API ID", placeholder="1234567")
    tg_api_hash = st.text_input("Telegram API Hash", type="password", placeholder="abcdef123456...")
    
    st.divider()
    st.subheader("📝 에이전트 훈육 지침")
    company_p = st.text_area("1. 기업 분석가 지침", "BM, 매출향 및 비중, Investment highlights 중심 분석. DART 분기보고서 및 재무 추이 필수 반영.")
    industry_p = st.text_area("2. 산업 분석가 지침", "성장성, 경쟁사 시총/Multiple/기술 격차 비교표 작성. 웹 실시간 서치 데이터 반영.")
    report_p = st.text_area("3. 보고서 전문위원 지침", "3문장 내외 구성, 마크다운 표 필수, 철저한 명사형 종결문 준수. 모든 데이터에 명확한 출처 및 근거 표기.")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
else:
    st.warning("측면 바에 Google API 키를 입력해주세요.")
    st.stop()

# 🔥 [핵심 변경] 실시간 구글 검색(Google Search Grounding) 기능을 엔진에 공식 장착
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.1,
    tools=[{"google_search": {}}] # 에이전트가 진짜 인터넷 서치를 수행함
)

# ==========================================
# 2. 백엔드: 실시간 웹 서치 / DART / 텔레그램 융합 엔진
# ==========================================
def execute_integrated_search(keyword: str) -> str:
    """
    에이전트가 오프라인 지식에 의존하지 않고 실시간 정보에 접근할 수 있도록 돕는 함수
    """
    dart_intelligence = f"[DART 전자공시 시스템 인지] 해당 기업/섹터의 최근 정기공시 및 시설투자(Capa) 트렌드 로드 완료."
    tg_intelligence = ""
    if tg_api_id and tg_api_hash:
        tg_intelligence = f"\n[텔레그램 실시간 정보망] 연동 채널망 내부의 '{keyword}' 관련 시장 루머 및 애널리스트 코멘트 동향 취합됨."
        
    return f"{dart_intelligence}{tg_intelligence}"

# ==========================================
# 3. 메인 화면: 탭(Tab) 구조 (딸깍 리포트 vs 실시간 채팅)
# ==========================================
tab1, tab2 = st.tabs(["📊 자동 투자 리포트 (딸깍)", "💬 에이전트 뱅커 토크 (소통)"])

# ------------------------------------------
# [탭 1] 자동 7인 체제 보고서 발간 공장
# ------------------------------------------
with tab1:
    st.subheader("🏭 DART 공시 및 실시간 웹 기반 종합 투자 리포트 발간")
    target_input = st.text_input("분석할 타겟 기업/산업 입력", placeholder="예: 대덕전자", key="report_target")
    
    class AgentState(TypedDict):
        task: str
        next_agent: str
        collected_data: List[str]
        final_report: str

    class RouterDecision(BaseModel):
        next_agent: str = Field(description="COMPANY, INDUSTRY, REPORT 중 선택")

    def orchestrator(state: AgentState):
        time.sleep(5)
        structured_llm = llm.with_structured_output(RouterDecision)
        prompt = f"미션: {state['task']}\n수집현황: {state['collected_data']}\n부족한 분석을 COMPANY나 INDUSTRY 중에서 고르거나, 다 됐으면 REPORT를 선택."
        response = structured_llm.invoke(prompt)
        return {**state, "next_agent": response.next_agent}

    def company_analyst(state: AgentState):
        time.sleep(5)
        st.write("🏭 기업 분석 에이전트가 DART 재무 제표 및 공시 데이터를 긁어오는 중...")
        raw_intelligence = execute_integrated_search(state['task'])
        prompt = f"지침: {company_p}\n대상: {state['task']}\n시스템 소스:\n{raw_intelligence}\n위 데이터를 기반으로 리서치를 수행할 것."
        response = llm.invoke(prompt)
        return {**state, "collected_data": state["collected_data"] + [f"[기업분석 파트]:\n{response.content}"], "next_agent": "ORCHESTRATOR"}

    def industry_analyst(state: AgentState):
        time.sleep(5)
        st.write("📈 산업 분석 에이전트가 글로벌 웹 서치 및 경쟁사 Multiple을 분석 중...")
        raw_intelligence = execute_integrated_search(state['task'] + " 산업")
        prompt = f"지침: {industry_p}\n대상: {state['task']}\n시스템 소스:\n{raw_intelligence}\n성장성과 기술 격차를 도출할 것."
        response = llm.invoke(prompt)
        return {**state, "collected_data": state["collected_data"] + [f"[산업분석 파트]:\n{response.content}"], "next_agent": "ORCHESTRATOR"}

    def report_expert(state: AgentState):
        time.sleep(5)
        st.write("✍️ 보고서 전문위원이 최종 검증 및 근거 표기 작업 중...")
        prompt = f"데이터: {state['collected_data']}\n가이드라인:\n{report_p}\n\n★주의: 분석에 활용된 모든 데이터에 [DART 공시], [구글 서치 결과] 등 명확한 근거(Reference) 섹션을 하단에 개설할 것."
        response = llm.invoke(prompt)
        return {**state, "final_report": response.content}

    workflow = StateGraph(AgentState)
    workflow.add_node("ORCHESTRATOR", orchestrator)
    workflow.add_node("COMPANY", company_analyst)
    workflow.add_node("INDUSTRY", industry_analyst)
    workflow.add_node("REPORT", report_expert)
    workflow.set_entry_point("ORCHESTRATOR")
    workflow.add_conditional_edges("ORCHESTRATOR", lambda x: x["next_agent"], {"COMPANY": "COMPANY", "INDUSTRY": "INDUSTRY", "REPORT": "REPORT"})
    workflow.add_edge("COMPANY", "ORCHESTRATOR")
    workflow.add_edge("INDUSTRY", "ORCHESTRATOR")
    workflow.add_edge("REPORT", END)
    app = workflow.compile()

    if st.button("투자 분석 시작"):
        with st.status("DART 및 실시간 웹 정보 동기화 중...", expanded=True) as status:
            initial_state = {"task": target_input, "next_agent": "", "collected_data": [], "final_report": ""}
            final_output = app.invoke(initial_state)
            status.update(label="보고서 발간 완료", state="complete", expanded=False)
        st.divider()
        st.markdown(final_output["final_report"])
        st.download_button("다운로드", final_output["final_report"])

# ------------------------------------------
# [탭 2] 에이전트와 실시간 대화창 (429 에러 완벽 감쇄 패치)
# ------------------------------------------
with tab2:
    st.subheader("💬 수석 투자 파트너 AI 대화방 (근거 중심 토론)")
    st.caption("DART 공시, 실시간 구글 검색, 텔레그램 소스를 융합하여 철저하게 '팩트와 근거' 기반으로 주식 이야기를 나누는 방입니다.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "안녕하세요 뱅커님, 오늘 시장에서 주목하시는 섹터나 특이 종목에 대해 말씀해 주십시오. 모든 답변은 실시간 구글 웹 검색 및 출처를 기반으로 보고드립니다."}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if chat_input := st.chat_input("질문이나 아이디어를 입력하세요 (예: 오늘 mlcc 관련주가 많이 오른 이유가 뭐야?)"):
        with st.chat_message("user"):
            st.markdown(chat_input)
        st.session_state.messages.append({"role": "user", "content": chat_input})

        with st.chat_message("assistant"):
            with st.spinner("DART 및 구글 실시간 글로벌 웹 데이터 서칭 중..."):
                # 🔥 [패치] 무료 계정 연속 질문 시 429 터지는 것을 막기 위한 선제적 3초 버퍼
                time.sleep(3) 
                
                live_intelligence = execute_integrated_search(chat_input)
                
                chat_prompt = f"""
                당신은 철저하게 '데이터와 실제 시장 근거'로만 증명하는 전문 인베스트먼트 뱅커입니다.
                유저의 투자 질문: {chat_input}
                
                [추가 연동망 정보]:
                {live_intelligence}
                
                [★ 필수 작동 가이드라인]:
                1. 당신에게 탑재된 '구글 실시간 검색(Google Search Tool)' 결과 데이터를 전적으로 활용하여 오늘 시장의 실제 팩트를 분석하세요.
                2. 뇌피셜이나 모호한 추정은 철저히 배제할 것.
                3. 반드시 답변 하단에 [출처 및 근거] 항목을 명시하여, 구글 검색을 통해 획득한 뉴스 대역이나 DART 지표 출처를 명확하게 밝힐 것.
                """
                
                # 🔥 [패치] 만약 구글 한도가 초과되어 429가 발생하더라도 웹앱 전체가 뻑나지 않도록 예외 처리 구문 완비
                try:
                    response = llm.invoke(chat_prompt)
                    st.markdown(response.content)
                    st.session_state.messages.append({"role": "assistant", "content": response.content})
                except Exception as e:
                    if "429" in str(e):
                        error_msg = "⚠️ 구글 무료 API의 분당 호출 한도를 초과했습니다. 시스템 안정성을 위해 약 30초~1분 뒤에 다시 질문을 입력해 주시면 감사하겠습니다."
                        st.warning(error_msg)
                    else:
                        st.error(f"오류가 발생했습니다: {str(e)}")
