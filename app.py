import streamlit as st
import os
from typing import TypedDict, List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ==========================================
# 1. 웹 UI 설정 (Streamlit)
# ==========================================
st.set_page_config(page_title="IB AI Research Hub", layout="wide")
st.title("🏢 AI 투자 리서치 본부 (7인 에이전트 시스템)")

with st.sidebar:
    st.header("⚙️ 설정 및 학습 센터")
    api_key = st.text_input("Google API Key", type="password")
    st.divider()
    st.subheader("📝 에이전트 지침 수정 (가르치기)")
    # 프롬프트 파일 읽기/쓰기 기능
    company_p = st.text_area("기업 분석가 지침", "매출 구조, 매출향, Investment Highlights를 명사형으로 분석.")
    industry_p = st.text_area("산업 분석가 지침", "성장성, 경쟁사 시총/Multiple/기술 격차 비교표 작성.")

# API 키 설정
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
else:
    st.warning("측면 바에 Google API 키를 입력해주세요.")
    st.stop()

# ==========================================
# 2. 에이전트 지능 및 지식(RAG) 설정
# ==========================================
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

# [학습 기능] knowledge_base 폴더의 PDF 읽기
def load_knowledge():
    if not os.path.exists("knowledge_base"):
        os.makedirs("knowledge_base")
    
    pdf_files = [f for f in os.listdir("knowledge_base") if f.endswith('.pdf')]
    if pdf_files:
        st.info(f"📚 {len(pdf_files)}개의 최신 리포트를 학습 데이터로 참조합니다.")
        # 간단한 문서 로딩 로직 (RAG 맛보기)
        # 실제 운영 시에는 여기서 Vector DB를 검색함
        return f"참고 파일 리스트: {pdf_files}"
    return "별도 학습 문서 없음 (기본 지식 사용)"

knowledge_context = load_knowledge()

# ==========================================
# 3. 랭그래프 에이전트 로직
# ==========================================
class AgentState(TypedDict):
    task: str
    next_agent: str
    collected_data: List[str]
    final_report: str

class RouterDecision(BaseModel):
    next_agent: str = Field(description="COMPANY, INDUSTRY, REPORT 중 선택")

def orchestrator(state: AgentState):
    structured_llm = llm.with_structured_output(RouterDecision)
    prompt = f"미션: {state['task']}\n수집현황: {state['collected_data']}\n부족한 분석을 COMPANY나 INDUSTRY 중에서 고르거나, 다 됐으면 REPORT를 선택."
    response = structured_llm.invoke(prompt)
    return {**state, "next_agent": response.next_agent}

def company_analyst(state: AgentState):
    st.write("🏭 기업 분석 에이전트 가동 중...")
    prompt = f"지침: {company_p}\n대상: {state['task']}\n참조지식: {knowledge_context}"
    response = llm.invoke(prompt)
    return {**state, "collected_data": state["collected_data"] + [f"[기업분석]: {response.content}"], "next_agent": "ORCHESTRATOR"}

def industry_analyst(state: AgentState):
    st.write("📈 산업 분석 에이전트 가동 중...")
    prompt = f"지침: {industry_p}\n대상: {state['task']}\n참조지식: {knowledge_context}"
    response = llm.invoke(prompt)
    return {**state, "collected_data": state["collected_data"] + [f"[산업분석]: {response.content}"], "next_agent": "ORCHESTRATOR"}

def report_expert(state: AgentState):
    st.write("✍️ 수석 뱅커 보고서 작성 중...")
    prompt = f"데이터: {state['collected_data']}\n가이드: 3문장 이내, 명사형 종결, 비교 표 필수 포함."
    response = llm.invoke(prompt)
    return {**state, "final_report": response.content}

# 그래프 조립
workflow = StateGraph(AgentState)
workflow.add_node("ORCHESTRATOR", orchestrator)
workflow.add_node("COMPANY", company_analyst)
workflow.add_node("INDUSTRY", industry_analyst)
workflow.add_node("REPORT", report_expert)
workflow.set_entry_point("ORCHESTRATOR")

workflow.add_conditional_edges("ORCHESTRATOR", lambda x: x["next_agent"], 
                              {"COMPANY": "COMPANY", "INDUSTRY": "INDUSTRY", "REPORT": "REPORT"})
workflow.add_edge("COMPANY", "ORCHESTRATOR")
workflow.add_edge("INDUSTRY", "ORCHESTRATOR")
workflow.add_edge("REPORT", END)
app = workflow.compile()

# ==========================================
# 4. 웹 UI 구동 화면
# ==========================================
target_input = st.text_input("분석할 타겟 기업/산업을 입력하세요", placeholder="예: 삼성전기 MLCC 사업")

if st.button("투자 분석 시작"):
    with st.status("에이전트 팀 협업 중...", expanded=True) as status:
        initial_state = {
            "task": target_input,
            "next_agent": "",
            "collected_data": [],
            "final_report": ""
        }
        final_output = app.invoke(initial_state)
        status.update(label="분석 완료!", state="complete", expanded=False)

    st.divider()
    st.subheader("📊 최종 투자 검토 보고서")
    st.markdown(final_output["final_report"])
    
    # 보고서 다운로드 버튼
    st.download_button("보고서 다운로드(TXT)", final_output["final_report"])