"""Decision execution and deterministic combination rules."""

from app.decisions.engine import DecisionEngineError, SystemOneProvider
from app.decisions.schemas import DecisionEvaluation, DecisionPlan


def evaluate_plan(state: str, plan: DecisionPlan, engine: SystemOneProvider) -> DecisionEvaluation:
    values = engine.evaluate(state, plan)
    if len(values) != len(plan.decisions) or {v.question_id for v in values} != {
        d.id for d in plan.decisions
    }:
        raise DecisionEngineError("Decision provider returned an incomplete result")
    outcome = None
    if plan.workflow == "eligibility":
        booleans = [bool(value.value) for value in values]
        eligible = all(booleans) if plan.combination == "all" else any(booleans)
        outcome = "eligible" if eligible else "not eligible"
    return DecisionEvaluation(decisions=values, outcome=outcome, combination=plan.combination)
