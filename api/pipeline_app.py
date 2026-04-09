"""Standalone FastAPI API layer for inference + grading + full pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

try:
    from ..core.config import get_config
    from ..core.exceptions import GradingError, InferenceError, PipelineError, WorkplaceEnvError
    from ..core.inference import AsyncInference, EnhancedInference, StandardInference
    from ..core.graders import CATEGORY_OPTIONS, RuleBasedRewardPolicy
    from ..core.logging_config import get_logger, setup_logging
except ImportError:  # pragma: no cover
    from core.config import get_config
    from core.exceptions import GradingError, InferenceError, PipelineError, WorkplaceEnvError
    from core.inference import AsyncInference, EnhancedInference, StandardInference
    from core.graders import CATEGORY_OPTIONS, RuleBasedRewardPolicy
    from core.logging_config import get_logger, setup_logging


setup_logging()
CFG = get_config()
LOGGER = get_logger(__name__)


class InferenceStrategy(str, Enum):
    standard = "standard"
    enhanced = "enhanced"
    async_strategy = "async"


class ActionPayload(BaseModel):
    action_type: Literal["classify", "reply", "escalate"]
    content: str = Field(min_length=1)


class InferRequest(BaseModel):
    email: str = Field(min_length=1, description="Customer email text")
    strategy: InferenceStrategy = InferenceStrategy.standard
    category_options: List[str] = Field(default_factory=lambda: list(CATEGORY_OPTIONS))
    scenario_difficulty: str = "easy"
    urgency: str = "medium"
    sentiment: str = "neutral"
    complexity_score: int = Field(default=2, ge=1, le=5)


class GradeRequest(BaseModel):
    action_type: Literal["classify", "reply", "escalate"]
    content: str = Field(min_length=1)
    actual_category: Literal["refund", "complaint", "query"]
    step_count: int = Field(ge=1, le=3)
    scenario_difficulty: str = "easy"
    min_reply_length: int = Field(default=30, ge=1)
    previous_actions: Dict[str, float] = Field(default_factory=dict)


class PipelineRequest(BaseModel):
    email: str = Field(min_length=1)
    actual_category: Literal["refund", "complaint", "query"]
    strategy: InferenceStrategy = InferenceStrategy.standard
    task: Optional[Literal["refund", "complaint", "query"]] = None
    scenario_difficulty: str = "easy"
    min_reply_length: int = Field(default=30, ge=1)


class ErrorPayload(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class InferResponse(BaseModel):
    score: float
    breakdown: Dict[str, Any]


class GradeResponse(BaseModel):
    score: float
    breakdown: Dict[str, Any]


class PipelineResponse(BaseModel):
    score: float
    breakdown: Dict[str, Any]


app = FastAPI(
    title="Workplace Env Pipeline API",
    version="1.0.0",
    description="Inference, grading, and end-to-end pipeline endpoints",
)


def _error_response(status_code: int, code: str, message: str, details: Optional[Dict[str, Any]] = None) -> JSONResponse:
    payload = {
        "success": False,
        "error": ErrorPayload(code=code, message=message, details=details).model_dump(),
    }
    return JSONResponse(status_code=status_code, content=payload)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    return _error_response(
        status_code=422,
        code="VALIDATION_ERROR",
        message="Invalid request payload",
        details={"errors": exc.errors()},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {"detail": str(exc.detail)}
    return _error_response(
        status_code=exc.status_code,
        code="HTTP_ERROR",
        message="Request failed",
        details=detail,
    )


@app.exception_handler(Exception)
async def general_exception_handler(_: Request, exc: Exception):  # pragma: no cover
    if isinstance(exc, WorkplaceEnvError):
        return _error_response(
            status_code=400,
            code=exc.code,
            message=str(exc),
            details=exc.details,
        )

    LOGGER.exception("Unhandled API exception")
    return _error_response(
        status_code=500,
        code="INTERNAL_ERROR",
        message="Unexpected server error",
        details={"exception": str(exc)},
    )


def _select_strategy(strategy: InferenceStrategy):
    LOGGER.debug("Selecting inference strategy: %s", strategy.value)
    if strategy == InferenceStrategy.standard:
        return StandardInference()
    if strategy == InferenceStrategy.enhanced:
        return EnhancedInference()
    return AsyncInference()


def _make_observation(req: InferRequest | PipelineRequest) -> Dict[str, Any]:
    category_options = list(getattr(req, "category_options", CATEGORY_OPTIONS))
    return {
        "email": req.email,
        "category_options": category_options,
        "scenario_difficulty": getattr(req, "scenario_difficulty", "easy"),
        "urgency": getattr(req, "urgency", "medium"),
        "sentiment": getattr(req, "sentiment", "neutral"),
        "complexity_score": getattr(req, "complexity_score", 2),
        "scenario_metadata": {"min_reply_length": getattr(req, "min_reply_length", 30)},
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"success": True, "data": {"status": "ok"}}


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    return {
        "success": True,
        "tasks": [
            {
                "name": "easy-triage",
                "description": "Clear single-intent emails — classify, reply, escalate",
                "actions": ["classify", "reply", "escalate"],
            },
            {
                "name": "medium-triage",
                "description": "Mixed-sentiment, ambiguous-intent emails",
                "actions": ["classify", "reply", "escalate"],
            },
            {
                "name": "hard-triage",
                "description": "Adversarial emails with sarcasm, multi-intent, threats",
                "actions": ["classify", "reply", "escalate"],
            },
        ],
    }


@app.post("/infer")
def infer(request: InferRequest) -> Dict[str, Any]:
    LOGGER.info("/infer request received with strategy=%s", request.strategy.value)
    strategy = _select_strategy(request.strategy)
    observation = _make_observation(request)

    try:
        actions = strategy.build_actions(observation)
    except Exception as exc:
        raise InferenceError("Inference strategy failed", details={"strategy": request.strategy.value, "exception": str(exc)}) from exc

    # Grade the inferred actions to produce an honest score
    policy = RuleBasedRewardPolicy()
    previous_actions: Dict[str, float] = {}
    cumulative = 0.0
    step_results: List[Dict[str, Any]] = []

    # Use the first category option as the ground truth for grading
    actual_category = observation.get("category_options", ["query"])[0]

    for idx, (action_type, content) in enumerate(actions, start=1):
        try:
            reward, step_breakdown = policy.calculate_step_reward(
                action_type=action_type,
                content=content,
                actual_category=actual_category,
                step_count=min(idx, 3),
                scenario_difficulty=request.scenario_difficulty,
                min_reply_length=observation.get("scenario_metadata", {}).get("min_reply_length", 30),
                previous_actions=previous_actions,
            )
        except Exception:
            reward = 0.0
            step_breakdown = {"error": "grading failed"}
        previous_actions[action_type] = reward
        cumulative += reward
        step_results.append({"step": idx, "action_type": action_type, "content": content, "score": reward})

    breakdown = {
        "strategy": request.strategy.value,
        "email": request.email,
        "action_count": len(actions),
        "actions": [{**s} for s in step_results],
        "observation": observation,
    }

    response = InferResponse(score=cumulative, breakdown=breakdown)
    return {"success": True, **response.model_dump()}


@app.post("/grade")
def grade(request: GradeRequest) -> Dict[str, Any]:
    LOGGER.info("/grade request received for action_type=%s", request.action_type)
    policy = RuleBasedRewardPolicy()
    try:
        reward, breakdown = policy.calculate_step_reward(
            action_type=request.action_type,
            content=request.content,
            actual_category=request.actual_category,
            step_count=request.step_count,
            scenario_difficulty=request.scenario_difficulty,
            min_reply_length=request.min_reply_length,
            previous_actions=request.previous_actions,
        )
    except Exception as exc:
        raise GradingError("Failed to grade action", details={"action_type": request.action_type, "exception": str(exc)}) from exc

    response = GradeResponse(score=reward, breakdown=breakdown)
    return {"success": True, **response.model_dump()}


@app.post("/pipeline")
def pipeline(request: PipelineRequest) -> Dict[str, Any]:
    LOGGER.info("/pipeline request received with strategy=%s", request.strategy.value)
    strategy = _select_strategy(request.strategy)
    policy = RuleBasedRewardPolicy()

    observation = _make_observation(request)
    try:
        actions = strategy.build_actions(observation)
    except Exception as exc:
        raise InferenceError("Failed to generate pipeline actions", details={"strategy": request.strategy.value, "exception": str(exc)}) from exc

    task_action_relevance: Dict[str, set[str]] = {
        "refund": {"classify", "reply"},
        "complaint": {"classify", "reply", "escalate"},
        "query": {"classify", "reply"},
    }

    if request.task is not None:
        allowed_actions = task_action_relevance.get(request.task, {"classify", "reply", "escalate"})
        actions = [(action_type, content) for action_type, content in actions if action_type in allowed_actions]

    previous_actions: Dict[str, float] = {}
    cumulative = 0.0
    step_results: List[Dict[str, Any]] = []

    for idx, (action_type, content) in enumerate(actions, start=1):
        try:
            reward, step_breakdown = policy.calculate_step_reward(
                action_type=action_type,
                content=content,
                actual_category=request.actual_category,
                step_count=min(idx, 3),
                scenario_difficulty=request.scenario_difficulty,
                min_reply_length=request.min_reply_length,
                previous_actions=previous_actions,
            )
        except Exception as exc:
            raise PipelineError(
                "Pipeline grading failed",
                details={"step": idx, "action_type": action_type, "exception": str(exc)},
            ) from exc
        previous_actions[action_type] = reward
        cumulative += reward
        step_results.append(
            {
                "step": idx,
                "action_type": action_type,
                "content": content,
                "score": reward,
                "breakdown": step_breakdown,
            }
        )

    breakdown = {
        "strategy": request.strategy.value,
        "email": request.email,
        "actual_category": request.actual_category,
        "steps": step_results,
        "action_rewards": previous_actions,
        "total_steps": len(step_results),
    }

    response = PipelineResponse(score=cumulative, breakdown=breakdown)
    return {"success": True, **response.model_dump()}


def main(host: str = CFG.api.host, port: int = CFG.api.pipeline_port):
    import uvicorn

    uvicorn.run("api.pipeline_app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
