import os
from typing import Annotated, TypedDict, List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# [필수] 본인의 구글 API 키를 넣어주세요.
os.environ["GOOGLE_API_KEY"] = "AIzaSyA5Rs2YqBESpnpTHj-m6wj_0Wbd2sf3_yQ" 

# ==========================================================
# 1. 공동 가방(State) 정의
# ==========================================================
class AgentState(TypedDict):
    task: str                  # 유저가 던진 미션 (타겟 기업)
    next_agent: str            # 관리자가 지정한 다음 일할 사람
    collected_data: List[str]  # 에이전트들이 쌓은 분석 정보
    final_report: str          # 최종 완성본 보고서

# 뇌 장착 (2.5-flash 모델 사용)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

# 관리자의 판단 양식
class RouterDecision(BaseModel):
    next_agent: str = Field(description="다음 에이전트 선택: 'COMPANY_ANALYSIS', 'INDUSTRY_ANALYSIS', 또는 양쪽 조사가 완료되면 'REPORT_EXPERT'")

# ==========================================================
# 2. 유저 맞춤형 3대 에이전트 정의
# ==========================================================

# [1] 총괄 관리자 (Orchestrator)
def orchestrator(state: AgentState):
    print("\n[🎯 총괄 관리자]: 가방 내 분석 자료를 검토 중입니다...")
    structured_llm = llm.with_structured_output(RouterDecision)
    
    prompt = f"""
    당신은 투자 심의를 총괄하는 수석 뱅커(Orchestrator)입니다.
    유저의 분석 요청: {state['task']}
    현재 수집된 데이터: {state['collected_data']}
    
    1. 타겟 기업의 수익 구조 및 Investment Highlights 조사 데이터가 없으면 -> 'COMPANY_ANALYSIS'
    2. 산업 성장성 및 글로벌 경쟁사 밸류에이션(시총/Multiple/기술) 비교 데이터가 없으면 -> 'INDUSTRY_ANALYSIS'
    3. 위 두 가지 심층 분석 자료가 가방에 모두 확보되었다면 -> 'REPORT_EXPERT'를 호출하여 최종 보고서 작성을 지시하세요.
    """
    response = structured_llm.invoke(prompt)
    return {**state, "next_agent": response.next_agent}


# [2] 기업 분석 에이전트 (Company Analysis)
def company_analyst(state: AgentState):
    print("[🏭 기업 분석가]: 타겟 기업의 비즈니스 모델 및 핵심 투자 하이라이트 분석 중...")
    
    prompt = f"""
    다음 타겟 기업(혹은 섹터)에 대해 투자자 관점에서 아래 핵심 사항을 심층 분석하여 서술형으로 가방에 담으세요.
    타겟 대상: {state['task']}
    
    [필수 포함 내용]
    1. BM 및 수익 구조 (어떻게 돈을 버는가)
    2. 매출처(매출향) 다변화 수준 및 비중 구성
    3. 해당 기업 고유의 핵심 Investment Highlights (투자 매력도)
    """
    response = llm.invoke(prompt)
    new_data = f"■ [기업분석 원천자료]\n{response.content}"
    return {
        **state, 
        "collected_data": state["collected_data"] + [new_data],
        "next_agent": "ORCHESTRATOR"
    }


# [3] 산업 분석 에이전트 (Industry Analysis)
def industry_analyst(state: AgentState):
    print("[📈 산업 분석가]: 전방 산업 성장성 및 경쟁사 Valuation Multiple 비교 분석 중...")
    
    prompt = f"""
    다음 타겟 산업(혹은 섹터)에 대해 아래 비교 프레임워크에 맞춰 정량적/정성적 데이터를 도출하여 가방에 담으세요.
    타겟 대상: {state['task']}
    
    [필수 포함 내용]
    1. 전방 산업의 향후 성장성 및 확장 모멘텀
    2. 글로벌 주요 경쟁사 라인업 및 시가총액 격차 현황
    3. 경쟁사별 Valuation Multiple (P/E, EV/EBITDA 등) 및 프리미엄/디스카운트 요인
    4. 경쟁사 간 핵심 기술 차별화 요소 및 진입장벽
    """
    response = llm.invoke(prompt)
    new_data = f"■ [산업분석 원천자료]\n{response.content}"
    return {
        **state, 
        "collected_data": state["collected_data"] + [new_data],
        "next_agent": "ORCHESTRATOR"
    }


# [4] 보고서 전문 에이전트 (Report Expert) - ★엄격한 포맷팅 적용
def report_expert(state: AgentState):
    print("[✍️ 보고서 전문위원]: 뱅커 스펙의 서면 보고서 양식으로 가공 중...")
    
    prompt = f"""
    당신은 까다로운 투자심의위원회를 만족시켜야 하는 대형 IB의 수석 커뮤니케이션 위원입니다. 
    아래 하위 팀원들이 조사해온 원천자료를 바탕으로, 유저가 제시한 **엄격한 서면 양식 가이드라인**을 100% 준수하여 정식 보고서를 작성하세요.
    
    [하위 팀원들의 분석 자료]
    {state['collected_data']}
    
    [★ 보고서 작성 가이드라인 - 필수 준수]
    1. 문체 제한: 모든 문장은 철저하게 명사형 어미(예: ~임, ~함, ~구축, ~지속, ~전망)로 끝맺음할 것. (~다, ~습니다 사용 절대 금지)
    2. 분량 제한: 각 대주제별 문단은 가독성을 위해 딱 3문장 내외로 콤팩트하게 구성할 것.
    3. 구조화: 
       - 1. 기업 분석 (수익구조, 매출향, Investment Highlights 요약)
       - 2. 산업 및 경쟁사 비교 (성장성, 시총/Multiple/기술 차이 비교표 포함)
    4. 표(Table) 삽입: 산업 및 경쟁사 비교 섹션에는 시총, Multiple, 기술력을 한눈에 비교할 수 있는 마크다운 표(Table)를 반드시 포함할 것.
    """
    response = llm.invoke(prompt)
    return {**state, "final_report": response.content}

# ==========================================================
# 3. 랭그래프(LangGraph) 제어판 조립
# ==========================================================
workflow = StateGraph(AgentState)

# 노드 등록
workflow.add_node("ORCHESTRATOR", orchestrator)
workflow.add_node("COMPANY_ANALYSIS", company_analyst)
workflow.add_node("INDUSTRY_ANALYSIS", industry_analyst)
workflow.add_node("REPORT_EXPERT", report_expert)

# 시작점 설정
workflow.set_entry_point("ORCHESTRATOR")

# 라우팅 규칙
def router_rule(state: AgentState):
    return state["next_agent"]

workflow.add_conditional_edges(
    "ORCHESTRATOR",
    router_rule,
    {
        "COMPANY_ANALYSIS": "COMPANY_ANALYSIS",
        "INDUSTRY_ANALYSIS": "INDUSTRY_ANALYSIS",
        "REPORT_EXPERT": "REPORT_EXPERT"
    }
)

# 회귀 경로 및 종료선 구축
workflow.add_edge("COMPANY_ANALYSIS", "ORCHESTRATOR")
workflow.add_edge("INDUSTRY_ANALYSIS", "ORCHESTRATOR")
workflow.add_edge("REPORT_EXPERT", END)

# 앱 컴파일
app = workflow.compile()

# ==========================================================
# 4. 실행 테스트 (타겟: 삼성전기 및 MLCC 섹터)
# ==========================================================
if __name__ == "__main__":
    initial_bag = {
        "task": "삼성전기 및 글로벌 MLCC 산업",
        "next_agent": "",
        "collected_data": [],
        "final_report": ""
    }
    
    print("🏢 [IB 특화 멀티 에이전트 시스템] 가동...")
    final_output = app.invoke(initial_bag)
    
    print("\n" + "="*60)
    print("📊 [정식 투자 검토 서면 보고서] 📊")
    print("="*60)
    print(final_output["final_report"])
    