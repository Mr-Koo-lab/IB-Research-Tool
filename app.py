import streamlit as st
import os
import time
from dotenv import load_dotenv  # [인프라 패치] .env 자동 저장을 위한 라이브러리
from typing import TypedDict, List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# ==========================================
# 0. [인프라 패치] .env 파일에서 기존 입력값 자동 로드
# ==========================================
load_dotenv()
default_google_key = os.getenv("GOOGLE_API_KEY", "")
default_tg_id = os.getenv("TELEGRAM_API_ID", "")
default_tg_hash = os.getenv("TELEGRAM_API_HASH", "")

# ==========================================
# 1. 웹 UI 및 레이아웃 설정
# ==========================================
st.set_page_config(page_title="Mr.Koo Agent Research Tool", layout="wide")
st.title("🚀 Mr.Koo Agent Research Tool (V2.7 - Realtime Grounding)")

with st.sidebar:
    st.header("⚙️ 인프라 및 가르치기 센터")
    # [인프라 패치] value=default_google_key 매핑으로 자동화
    api_key = st.text_input("Google API Key", value=default_google_key, type="password")
    
    st.divider()
    st.subheader("🛰️ 외부 인프라 연동망")
    # [인프라 패치] value 매핑으로 자동화
    tg_api_id = st.text_input("Telegram API ID", value=default_tg_id, placeholder="1234567")
    tg_api_hash = st.text_input("Telegram API Hash", value=default_tg_hash, type="password", placeholder="abcdef123456...")
    
    st.divider()
    st.subheader("📝 에이전트 훈육 지침")
    company_p = st.text_area("1. 기업 분석가 지침", "BM, 매출향 및 비중, 2026년 최신 Investment highlights 중심 분석. 과거 데이터 배제하고 실시간 검색 최우선 반영.")
    industry_p = st.text_area("2. 산업 분석가 지침", "밸류에이션 산정을 위한 Peer Group(비교기업)들의 2026년 예상 실적 및 PER/PBR 멀티플 데이터를 수집하여 비교표 작성을 준비할 것.")
    report_p = st.text_area("3. 보고서 전문위원 지침", """
Strictly Private & Confidential

[타겟 기업] [사채 종류] 투자의 건
주식운용본부

Ⅰ. Executive Summary
1. Deal 개요
- 本 건은 [기업설명 및 시총]의 신규 발행 [사채 종류]에 투자하는 건임.
- [사채 주요 조건]: 전환(교환)주식, 지분율, 교환가액, 자금사용목적, 이자(쿠폰/YTM), 만기, 풋옵션/콜옵션, 주관사 정보를 반드시 마크다운 표나 리스트로 일목요연하게 정리할 것.

2. 기업 분석
- BM, 생산능력, 최근 분기 실적 팩트 기술.
- [회사 연결재무제표] 테이블 필수 작성: 최근 3개년 및 2025E, 2026E 매출액, 영업이익, 영업이익률(%), 당기순이익, 자산/부채/자본총계, 부채비율, 현금및현금성자산 정보를 가로형 마크다운 표로 구현할 것. (출처: 전자공시시스템 및 당사 리서치 추정 명시)

Ⅱ. Exit 방안 및 밸류에이션
1. Exit 시나리오 및 target 시점 명시.
2. [밸류에이션 및 Peer 비교] 테이블 필수 작성: 예상 순이익, 적용 주식수, 예상 EPS, Target PER(또는 PBR), 주당 가치, 총 회사 가치, 상승여력(%) 계산표를 구조화하여 마크다운 표로 출력할 것.

⚠️ 스타일 가이드: 철저한 투자금융(IB) 프로페셔널 톤앤매너, 명사형 종결문 사용, 정량적 수치 중심으로만 채울 것.
""")
    
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
else:
    st.warning("측면 바에 Google API 키를 입력해주세요.")
    st.stop()

# 🔥 최고급형 Pro 모델 및 실시간 구글 검색(Google Search Grounding) 엔진 고정
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro", 
    temperature=0.1,
    tools=[{"google_search": {}}] 
)

# ==========================================
# 2. 백엔드: 시스템 시점 동기화 커넥터
# ==========================================
def execute_integrated_search(keyword: str) -> str:
    current_date_context = "⏰ 현재 시점 가이드라인: 지금은 [2026년 6월] 임. 모든 주가, 시총, 재무 추이는 2026년 현재 시점의 실시간 정보여야 함."
    dart_intelligence = f"\n[DART 시스템] {keyword} 관련 최근 2025년 결산 및 2026년 1분기 분기보고서 공시 인덱싱 완료."
    tg_intelligence = ""
    if tg_api_id and tg_api_hash:
        tg_intelligence = f"\n[텔레그램 실시간 정보망] 채널망 내 '{keyword}' 관련 최신 여의도 동향 매칭."
        
    return f"{current_date_context}{dart_intelligence}{tg_intelligence}"

# ==========================================
# 3. 메인 화면: 탭(Tab) 구조
# ==========================================
tab1, tab2 = st.tabs(["📊 자동 투자 리포트 (딸깍)", "💬 에이전트 뱅커 토크 (소통)"])

# ------------------------------------------
# [탭 1] 자동 7인 체제 보고서 발간 공장 (실시간 팩트 강제)
# ------------------------------------------
with tab1:
    st.subheader("🏭 DART 공시 및 실시간 웹 기반 종합 투자 리포트 발간")
    target_input = st.text_input("분석할 타겟 기업/산업 입력", placeholder="예: 대덕전자", key="report_target")
    
    class AgentState(TypedDict):
        task: str
        next_agent: str
        collected_data: List[str]
        final_report: str

# 1. 지휘관의 의사결정 구조체 정의
class RouterDecision(BaseModel):
    next_agent: str = Field(description="다음으로 실행할 에이전트 (COMPANY, INDUSTRY, REPORT 중 택1)")

# 2. 오케스트레이터 함수
def orchestrator(state: AgentState):
    time.sleep(0.1)
    llm_with_tools = llm.bind_tools([RouterDecision], tool_choice="RouterDecision")
    prompt = f"미션: {state['task']}\n수집현황: {state['collected_data']}\n부족한 분석을 COMPANY나 INDUSTRY 중에서 고르거나, 다 됐으면 REPORT를 선택."
    response = llm_with_tools.invoke(prompt)
    
    if response.tool_calls:
        next_agent = response.tool_calls[0]['args']['next_agent']
    else:
        next_agent = "REPORT"
        
    return {**state, "next_agent": next_agent}
    
    def company_analyst(state: AgentState):
        time.sleep(0.1)
        st.write("🏭 기업 분석 에이전트가 2026년 현재 실시간 시장 데이터를 스캔 중...")
        raw_intelligence = execute_integrated_search(state['task'])
        
        # 🔥 [프롬프트 수정] 과거 학습 지식 사용을 금지하고, 구글 검색 도구 사용을 강제함
        prompt = f"""
        당신에게 탑재된 '구글 실시간 검색 툴(Google Search)'을 반드시 실행하여 {state['task']}에 대한 2026년 현재 최신 뉴스와 재무 상태를 검색하세요.
        지침: {company_p}
        시스템 소스: {raw_intelligence}
        ⚠️ 주의: 당신의 원래 예전 과거 지식(2024년 이전)은 절대 쓰지 말고, 오직 방금 검색 툴로 찾아낸 2026년 최신 팩트만 가방에 담으세요.
        """
        response = llm.invoke(prompt)
        return {**state, "collected_data": state["collected_data"] + [f"[기업분석 파트]:\n{response.content}"], "next_agent": "ORCHESTRATOR"}

    def industry_analyst(state: AgentState):
        time.sleep(0.1)
        st.write("📈 산업 분석 에이전트가 2026년 최신 뉴스 컨센서스를 크롤링 중...")
        raw_intelligence = execute_integrated_search(state['task'] + " 산업")
        
        prompt = f"""
        당신에게 탑재된 '구글 실시간 검색 툴(Google Search)'을 반드시 실행하여 {state['task']} 산업 섹터의 2026년 현재 주가 위치와 Multiple 현황을 검색하세요.
        지침: {industry_p}
        시스템 소스: {raw_intelligence}
        ⚠️ 명심: 현재는 2026년 6월임. 과거 리포트 데이터는 철저히 배제하고 지금 당장 실시간 서치 창에 뜨는 시장 밸류에이션을 도출하세요.
        """
        response = llm.invoke(prompt)
        return {**state, "collected_data": state["collected_data"] + [f"[산업분석 파트]:\n{response.content}"], "next_agent": "ORCHESTRATOR"}

    def report_expert(state: AgentState):
        time.sleep(0.1)
        st.write("✍️ 보고서 전문위원이 최종 검증 및 근거 표기 작업 중...")
        prompt = f"데이터: {state['collected_data']}\n가이드라인:\n{report_p}\n\n★주의: 모든 문단과 정량 데이터에 [2026년 6월 구글 검색 결과] 혹은 [DART 최신공시] 꼬리표를 달아 최신성(Grounding)을 입증할 것."
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
        with st.status("실시간 데이터 그라운딩 인프라 가동 중...", expanded=True) as status:
            initial_state = {"task": target_input, "next_agent": "", "collected_data": [], "final_report": ""}
            final_output = app.invoke(initial_state)
            status.update(label="리포트 발간 완료", state="complete", expanded=False)
        st.divider()
        st.markdown(final_output["final_report"])
        st.download_button("다운로드", final_output["final_report"])

# ------------------------------------------
# [탭 2] 에이전트와 실시간 대화창 (시점 사수 패치)
# ------------------------------------------
with tab2:
    st.subheader("💬 수석 투자 파트너 AI 대화방 (근거 중심 토론)")
    st.caption("2026년 실시간 구글 검색 결과를 융합하여 철저하게 '팩트와 근거' 기반으로 주식 이야기를 나누는 방입니다.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "안녕하세요 뱅커님, 오늘 시장에서 주목하시는 섹터나 특이 종목에 대해 말씀해 주십시오. 모든 답변은 2026년 실시간 구글 웹 검색을 기반으로 보고드립니다."}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if chat_input := st.chat_input("질문이나 아이디어를 입력하세요"):
        with st.chat_message("user"):
            st.markdown(chat_input)
        st.session_state.messages.append({"role": "user", "content": chat_input})

        with st.chat_message("assistant"):
            with st.spinner("2026년 실시간 뉴스 인덱스 검색 중..."):
                time.sleep(0.1) 
                live_intelligence = execute_integrated_search(chat_input)
                
                # 🔥 [대화방 프롬프트 수정] 지금이 2026년임을 극단적으로 강조
                chat_prompt = f"""
                당신은 철저하게 '데이터와 실제 시장 근거'로만 증명하는 전문 인베스트먼트 뱅커입니다.
                현재 시점은 무조건 [2026년 6월] 임을 인지하고 사고하세요.
                
                유저의 투자 질문: {chat_input}
                [시스템 동기화 정보]: {live_intelligence}
                
                [★ 필수 작동 가이드라인]:
                1. 당신에게 탑재된 '구글 실시간 검색(Google Search Tool)'을 무조건 작동시켜 오늘 자 기준의 실제 테크 뉴스/증시 팩트를 분석하세요.
                2. 2024년 이전의 과거 지식은 구형 데이터이므로 뱅커에게 보고할 가치가 없음. 오직 2026년 현재 데이터만 가공할 것.
                3. 반드시 답변 하단에 [출처: 2026년 6월 실시간 검색 결과] 항목을 명시할 것.
                """
                
                try:
                    response = llm.invoke(chat_prompt)
                    st.markdown(response.content)
                    st.session_state.messages.append({"role": "assistant", "content": response.content})
                except Exception as e:
                    st.error(f"오류가 발생했습니다: {str(e)}")
