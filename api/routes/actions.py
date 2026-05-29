"""Action rules management endpoints."""

from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.agent.action_engine import ActionEngine
from src.models.confidence import ConfidenceReport
from src.models.extraction_result import ExtractionMode, ExtractionResult

router = APIRouter()

# In-memory rule storage for dynamic rule management
_dynamic_rules: Dict[str, Dict[str, Any]] = {}


class RuleCreate(BaseModel):
    """Request model for creating an action rule."""

    name: str = Field(..., description="Rule name")
    trigger_condition: str = Field(..., description="Trigger condition expression")
    action_type: str = Field(default="webhook", description="Action type")
    action_endpoint: str = Field(default="", description="Action endpoint/target")
    priority: str = Field(default="medium", description="Priority level")


class EvaluateRequest(BaseModel):
    """Request model for manually evaluating rules against an extraction result."""

    document_id: str = Field(..., description="Document ID")
    extraction_mode: str = Field(default="local", description="Extraction mode used")
    entities: Dict[str, Any] = Field(default_factory=dict, description="Extracted entities")
    overall_confidence: float = Field(default=0.8, description="Overall confidence")


@router.get("/rules")
async def list_rules():
    """List all configured action rules (from YAML config and dynamic)."""
    engine = ActionEngine()
    config_rules = []
    for rule in engine._rules:
        config_rules.append({
            "id": rule.get("name", "unnamed"),
            "name": rule.get("name", "unnamed"),
            "trigger": rule.get("trigger", {}),
            "action": rule.get("action", {}),
            "source": "config",
        })

    dynamic_rules = [
        {"id": rule_id, **rule_data, "source": "dynamic"}
        for rule_id, rule_data in _dynamic_rules.items()
    ]

    return {"rules": config_rules + dynamic_rules}


@router.post("/rules")
async def create_rule(rule: RuleCreate):
    """Add a new dynamic action rule."""
    rule_id = str(uuid4())
    _dynamic_rules[rule_id] = {
        "name": rule.name,
        "trigger": {"condition": rule.trigger_condition},
        "action": {
            "type": rule.action_type,
            "endpoint": rule.action_endpoint,
            "priority": rule.priority,
        },
    }
    return {"id": rule_id, "name": rule.name, "status": "created"}


@router.put("/rules/{rule_id}")
async def update_rule(rule_id: str, rule: RuleCreate):
    """Update an existing dynamic action rule."""
    if rule_id not in _dynamic_rules:
        raise HTTPException(status_code=404, detail=f"Rule not found: {rule_id}")

    _dynamic_rules[rule_id] = {
        "name": rule.name,
        "trigger": {"condition": rule.trigger_condition},
        "action": {
            "type": rule.action_type,
            "endpoint": rule.action_endpoint,
            "priority": rule.priority,
        },
    }
    return {"id": rule_id, "name": rule.name, "status": "updated"}


@router.delete("/rules/{rule_id}")
async def delete_rule(rule_id: str):
    """Remove a dynamic action rule."""
    if rule_id not in _dynamic_rules:
        raise HTTPException(status_code=404, detail=f"Rule not found: {rule_id}")

    del _dynamic_rules[rule_id]
    return {"id": rule_id, "status": "deleted"}


@router.post("/evaluate")
async def evaluate_rules(request: EvaluateRequest):
    """Manually evaluate action rules against a given extraction result."""
    confidence_report = ConfidenceReport(
        overall_confidence=request.overall_confidence,
        threshold=0.7,
    )

    result = ExtractionResult(
        document_id=request.document_id,
        extraction_mode=ExtractionMode(request.extraction_mode),
        entities=request.entities,
        confidence_report=confidence_report,
        raw_text="",
        processing_time_ms=0.0,
    )

    engine = ActionEngine()
    triggered_actions = engine.evaluate_rules(result)

    return {
        "document_id": request.document_id,
        "rules_evaluated": len(engine._rules),
        "actions_triggered": len(triggered_actions),
        "actions": triggered_actions,
    }
