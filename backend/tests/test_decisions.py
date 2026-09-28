"""Phase 7 structured decision tests; all providers are deterministic fakes."""

import pytest
from pydantic import ValidationError

from app.decisions.engine import DecisionEngineError, LayaDecisionEngine
from app.decisions.schemas import DecisionPlan, DecisionQuestion
from app.decisions.service import evaluate_plan
from app.decisions.engine import get_decision_engine
from app.llm.provider import get_llm_provider
from app.main import app
from tests.rag_helpers import GPA_TEXT, LEAVE_TEXT, register_org, upload_pdf


class MockLLM:
    def __init__(self, plan):
        self.plan = plan
        self.calls = []

    def plan_decisions(self, question, context):
        self.calls.append(("plan", question, context))
        return self.plan

    def generate_answer(self, question, context, decision_summary=None):
        self.calls.append(("answer", question, context, decision_summary))
        return "Based on the policy, the applicant is not eligible."


class FakeRouter:
    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        return {"answers": self.answers}


def eligibility_plan(combination="all"):
    return DecisionPlan.model_validate({
        "requires_decisions": True,
        "workflow": "eligibility",
        "combination": combination,
        "decisions": [
            {"id": "meets_gpa", "question": "Does the GPA meet the minimum?", "decision_type": "boolean"},
            {"id": "meets_year", "question": "Is the student in an eligible year?", "decision_type": "boolean"},
        ],
    })


def test_valid_decision_types_and_invalid_criteria():
    assert DecisionQuestion(id="select", question="Pick one", decision_type="choice", criteria={"a": "A", "b": "B"})
    assert DecisionQuestion(id="classify", question="Classify", decision_type="classification", criteria={"yes": "Yes", "no": "No"})
    assert DecisionQuestion(id="score", question="Score", decision_type="score", criteria=["low", "high"])
    with pytest.raises(ValidationError):
        DecisionQuestion(id="bad", question="Pick", decision_type="choice", criteria={"only": "one"})
    with pytest.raises(ValidationError):
        DecisionQuestion(id="bad", question="Score", decision_type="score", criteria={"a": "A", "b": "B"})


def test_invalid_workflow_inputs_rejected():
    with pytest.raises(ValidationError):
        DecisionPlan.model_validate({"requires_decisions": True, "workflow": "eligibility", "combination": "all", "decisions": [
            {"id": "one", "question": "Only one?", "decision_type": "boolean"}
        ]})
    with pytest.raises(ValidationError):
        DecisionPlan.model_validate({"requires_decisions": True, "workflow": "classification", "decisions": [
            {"id": "one", "question": "Score?", "decision_type": "score", "criteria": ["low", "high"]}
        ]})
    with pytest.raises(ValidationError):
        DecisionPlan.model_validate({"requires_decisions": False, "decisions": [
            {"id": "one", "question": "Boolean?", "decision_type": "boolean"}
        ]})


def test_laya_adapter_maps_all_assignment_decision_types():
    plan = DecisionPlan.model_validate({"requires_decisions": True, "workflow": "classification", "decisions": [
        {"id": "choice", "question": "Choose", "decision_type": "choice", "criteria": {"a": "A", "b": "B"}},
        {"id": "class", "question": "Classify", "decision_type": "classification", "criteria": {"yes": "Yes", "no": "No"}},
        {"id": "score", "question": "Rate", "decision_type": "score", "criteria": ["low", "middle", "high"]},
        {"id": "bool", "question": "Pass?", "decision_type": "boolean"},
    ]})
    router = FakeRouter({"choice": {"choice": "a"}, "class": {"choice": "yes"},
                         "score": {"score": 2}, "bool": {"noul": 0.9}})
    engine = LayaDecisionEngine(router)
    results = engine.evaluate("grounded policy state", plan)
    assert router.calls[0][1]["bool"] == {"type": "noul", "instructions": "Pass?"}
    assert router.calls[0][1]["class"]["type"] == "choice"
    assert router.calls[0][1]["class"]["instructions"] == "Classify"
    assert [(r.decision_type, r.value) for r in results] == [
        ("choice", "a"), ("classification", "yes"), ("score", 2), ("boolean", True)
    ]
    assert results[-1].probability == 0.9


@pytest.mark.parametrize(("combination", "answers", "expected"), [
    ("all", {"meets_gpa": {"noul": 0.9}, "meets_year": {"noul": 0.1}}, "not eligible"),
    ("any", {"meets_gpa": {"noul": 0.9}, "meets_year": {"noul": 0.1}}, "eligible"),
])
def test_eligibility_combination_behavior(combination, answers, expected):
    plan = eligibility_plan(combination)
    evaluation = evaluate_plan("policy text", plan, LayaDecisionEngine(FakeRouter(answers)))
    assert evaluation.outcome == expected
    assert combination in evaluation.summary()


@pytest.mark.parametrize("answers", [
    {"meets_gpa": {"noul": 1.2}, "meets_year": {"noul": 0.5}},
    {"meets_gpa": {"noul": 0.9}},
])
def test_bad_laya_output_fails_closed(answers):
    with pytest.raises(DecisionEngineError):
        LayaDecisionEngine(FakeRouter(answers)).evaluate("state", eligibility_plan())


def test_combined_rag_and_system1_flow_is_grounded_and_cited(
    client, tmp_storage, embedding_provider
):
    org = register_org(client, "phase7-flow@example.com", "Decision Org")
    document = upload_pdf(client, org, [GPA_TEXT], filename="Scholarship.pdf")
    assistant = client.post("/api/assistant", headers=org.headers).json()
    llm = MockLLM(eligibility_plan())
    router = FakeRouter({"meets_gpa": {"noul": 0.9}, "meets_year": {"noul": 0.1}})
    app.dependency_overrides[get_llm_provider] = lambda: llm
    app.dependency_overrides[get_decision_engine] = lambda: LayaDecisionEngine(router)

    response = client.post("/api/ask", json={
        "assistant_token": assistant["assistant_token"],
        "question": "Does scholarship eligibility require a minimum GPA of 3.0?",
    })

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["decision_outcome"] == "not eligible"
    assert [decision["value"] for decision in body["decisions"]] == [True, False]
    assert body["citations"][0]["document_id"] == document["id"]
    assert GPA_TEXT in router.calls[0][0]
    assert "Structured System 1 decisions" in llm.calls[-1][3]


def test_structured_flow_cannot_read_another_tenant_even_with_forged_org_id(
    client, tmp_storage, embedding_provider
):
    org_a = register_org(client, "phase7-private-a@example.com", "Private A")
    org_b = register_org(client, "phase7-private-b@example.com", "Private B")
    upload_pdf(client, org_a, [GPA_TEXT], filename="A_Private.pdf")
    document_b = upload_pdf(client, org_b, [LEAVE_TEXT], filename="B_Private.pdf")
    assistant_b = client.post("/api/assistant", headers=org_b.headers).json()
    llm = MockLLM(eligibility_plan())
    router = FakeRouter({"meets_gpa": {"noul": 0.6}, "meets_year": {"noul": 0.7}})
    app.dependency_overrides[get_llm_provider] = lambda: llm
    app.dependency_overrides[get_decision_engine] = lambda: LayaDecisionEngine(router)

    response = client.post("/api/ask", json={
        "assistant_token": assistant_b["assistant_token"],
        "organization_id": org_a.organization_id,
        "question": LEAVE_TEXT,
    })

    assert response.status_code == 200, response.text
    assert "A_Private.pdf" not in router.calls[0][0]
    assert "B_Private.pdf" in router.calls[0][0]
    assert all(source["document_id"] == document_b["id"] for source in response.json()["citations"])


def test_missing_laya_is_safe_for_structured_decision(client, tmp_storage, embedding_provider):
    org = register_org(client, "phase7-no-laya@example.com", "No Laya")
    upload_pdf(client, org, [GPA_TEXT])
    assistant = client.post("/api/assistant", headers=org.headers).json()
    app.dependency_overrides[get_llm_provider] = lambda: MockLLM(eligibility_plan())
    app.dependency_overrides[get_decision_engine] = lambda: None

    response = client.post("/api/ask", json={
        "assistant_token": assistant["assistant_token"],
        "question": "Does scholarship eligibility require a minimum GPA of 3.0?",
    })
    assert response.status_code == 503
    assert "Structured decision service" in response.json()["detail"]
