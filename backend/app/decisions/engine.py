"""Independent adapter for the Laya System 1 decision engine."""

import logging
from typing import Protocol

from app.core.config import settings
from app.decisions.schemas import DecisionPlan, DecisionResult

logger = logging.getLogger(__name__)


class DecisionEngineError(Exception):
    """Decision provider is missing, failed, or returned unusable output."""


class SystemOneProvider(Protocol):
    def evaluate(self, state: str, plan: DecisionPlan) -> list[DecisionResult]: ...


class LayaDecisionEngine:
    """Translate assignment decision types to Laya's choice/score/noul API."""

    def __init__(self, router=None):
        if router is None:
            try:
                from laya import Router

                router = Router(device=settings.laya_device)
            except Exception as exc:
                raise DecisionEngineError("Laya decision engine is unavailable") from exc
        self._router = router

    def evaluate(self, state: str, plan: DecisionPlan) -> list[DecisionResult]:
        questions: dict[str, dict] = {}
        for item in plan.decisions:
            key = {"boolean": "noul", "classification": "choice"}.get(item.decision_type, item.decision_type)
            spec: dict = {"type": key, "instructions": item.question}
            if item.criteria is not None:
                spec["criteria"] = item.criteria
            questions[item.id] = spec
        try:
            response = self._router.predict(state, questions)
            answers = response["answers"]
            if not isinstance(answers, dict):
                raise ValueError("answers is not an object")
            results = []
            for item in plan.decisions:
                key = {"boolean": "noul", "classification": "choice"}.get(item.decision_type, item.decision_type)
                raw = answers[item.id][key]
                probability = None
                if item.decision_type == "boolean":
                    probability = float(raw)
                    if not 0 <= probability <= 1:
                        raise ValueError("boolean probability outside [0, 1]")
                    value: str | bool | int | float = probability >= 0.5
                elif item.decision_type in ("choice", "classification"):
                    if not isinstance(raw, str) or raw not in item.criteria:
                        raise ValueError("choice result does not match a declared label")
                    value = raw
                else:
                    from numbers import Real

                    if not isinstance(raw, Real) or isinstance(raw, bool):
                        raise ValueError("score result must be numeric")
                    value = float(raw)
                    if not 0 <= value < len(item.criteria):
                        raise ValueError("score result outside the declared ordinal range")
                results.append(DecisionResult(
                    question_id=item.id, question=item.question,
                    decision_type=item.decision_type, value=value, probability=probability,
                ))
            return results
        except DecisionEngineError:
            raise
        except Exception as exc:
            logger.exception("Laya decision evaluation failed")
            raise DecisionEngineError("Laya decision evaluation failed") from exc


_engine: SystemOneProvider | None = None
_engine_resolved = False


def get_decision_engine() -> SystemOneProvider | None:
    """Lazily initialize Laya; missing package/model support never blocks app boot."""
    global _engine, _engine_resolved
    if not _engine_resolved:
        try:
            _engine = LayaDecisionEngine()
        except DecisionEngineError:
            logger.info("Laya decision engine unavailable; structured questions are disabled")
            _engine = None
        _engine_resolved = True
    return _engine
