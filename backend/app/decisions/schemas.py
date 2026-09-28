"""Validated contracts for LLM-planned, System 1 evaluated decisions."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DecisionType = Literal["choice", "boolean", "score", "classification"]


class DecisionQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=40, pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$")
    question: str = Field(min_length=1, max_length=500)
    decision_type: DecisionType
    # choice/classification label descriptions; score ordinal labels.
    criteria: dict[str, str] | list[str] | None = None

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("decision question must not be blank")
        return value

    @model_validator(mode="after")
    def validate_criteria(self):
        if self.decision_type in ("choice", "classification"):
            if not isinstance(self.criteria, dict) or len(self.criteria) < 2:
                raise ValueError("choice and classification require at least two labeled criteria")
            if any(not label.strip() or not description.strip() for label, description in self.criteria.items()):
                raise ValueError("criteria labels and descriptions must not be blank")
        elif self.decision_type == "score":
            if not isinstance(self.criteria, list) or len(self.criteria) < 2:
                raise ValueError("score requires at least two ordered criteria")
            if any(not label.strip() for label in self.criteria):
                raise ValueError("score criteria must not be blank")
        elif self.criteria is not None:
            raise ValueError("boolean decisions do not accept criteria")
        return self


class DecisionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requires_decisions: bool
    workflow: Literal["eligibility", "classification", "scoring"] | None = None
    combination: Literal["all", "any"] | None = None
    decisions: list[DecisionQuestion] = Field(default_factory=list, max_length=6)

    @model_validator(mode="after")
    def validate_plan(self):
        if not self.requires_decisions:
            if self.workflow is not None or self.combination is not None or self.decisions:
                raise ValueError("a plan without decisions cannot define a workflow")
            return self
        if not self.workflow or not self.decisions:
            raise ValueError("decision plans require a workflow and at least one decision")
        ids = [decision.id for decision in self.decisions]
        if len(ids) != len(set(ids)):
            raise ValueError("decision IDs must be unique")
        if self.workflow == "eligibility":
            if self.combination not in ("all", "any"):
                raise ValueError("eligibility plans require all/any combination behavior")
            if len(self.decisions) < 2 or any(d.decision_type != "boolean" for d in self.decisions):
                raise ValueError("eligibility requires at least two independent boolean decisions")
        elif self.combination is not None:
            raise ValueError("combination is only valid for eligibility plans")
        elif self.workflow == "classification" and not any(
            d.decision_type in ("choice", "classification") for d in self.decisions
        ):
            raise ValueError("classification workflow requires a choice decision")
        elif self.workflow == "scoring" and not any(d.decision_type == "score" for d in self.decisions):
            raise ValueError("scoring workflow requires a score decision")
        return self


class DecisionResult(BaseModel):
    question_id: str
    question: str
    decision_type: DecisionType
    value: str | bool | int | float
    probability: float | None = None


class DecisionEvaluation(BaseModel):
    decisions: list[DecisionResult]
    outcome: str | None = None
    combination: Literal["all", "any"] | None = None

    def summary(self) -> str:
        lines = [f"- {item.question}: {item.value}" for item in self.decisions]
        if self.outcome is not None:
            lines.append(f"Combined outcome ({self.combination} conditions): {self.outcome}")
        return "Structured System 1 decisions:\n" + "\n".join(lines)
