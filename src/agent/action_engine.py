"""Action engine for evaluating rules against extraction results."""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import yaml

from src.models.extraction_result import ExtractionResult

logger = logging.getLogger(__name__)


class Action:
    """Represents a triggered action."""

    def __init__(
        self,
        action_type: str,
        target: str,
        payload: Dict[str, Any],
        triggered_by_rule: str,
        priority: str = "medium",
    ) -> None:
        """Initialize an action.

        Args:
            action_type: Type of action (webhook, email, route, flag).
            target: Target endpoint or destination.
            payload: Action payload data.
            triggered_by_rule: Name of the rule that triggered this action.
            priority: Action priority (low, medium, high, critical).
        """
        self.id = str(uuid4())
        self.action_type = action_type
        self.target = target
        self.payload = payload
        self.triggered_by_rule = triggered_by_rule
        self.priority = priority
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert action to a dictionary representation.

        Returns:
            Dictionary with action details.
        """
        return {
            "id": self.id,
            "type": self.action_type,
            "target": self.target,
            "payload": self.payload,
            "triggered_by_rule": self.triggered_by_rule,
            "priority": self.priority,
            "timestamp": self.timestamp.isoformat(),
        }


class ActionEngine:
    """Engine for evaluating action rules against extraction results.

    Loads rules from config/actions.yaml and evaluates them against
    extraction results to determine which actions should be triggered.
    Supports condition operators: gt, lt, eq, contains, exists.
    """

    def __init__(self, rules_path: Optional[str] = None) -> None:
        """Initialize the action engine.

        Args:
            rules_path: Path to the actions YAML file. Defaults to config/actions.yaml.
        """
        if rules_path is None:
            rules_path = str(
                Path(__file__).parent.parent.parent / "config" / "actions.yaml"
            )
        self._rules_path = rules_path
        self._rules = self._load_rules()

    def _load_rules(self) -> List[Dict[str, Any]]:
        """Load action rules from the YAML configuration file.

        Returns:
            List of rule definitions.
        """
        path = Path(self._rules_path)
        if not path.exists():
            logger.warning(f"Actions config not found: {self._rules_path}")
            return []

        try:
            with open(path, "r") as f:
                config = yaml.safe_load(f) or {}
            return config.get("actions", [])
        except Exception as e:
            logger.error(f"Failed to load action rules: {e}")
            return []

    def reload_rules(self) -> None:
        """Reload rules from the YAML file for dynamic rule updates."""
        self._rules = self._load_rules()

    def evaluate_rules(self, extraction_result: ExtractionResult) -> List[Dict[str, Any]]:
        """Evaluate all rules against an extraction result.

        Args:
            extraction_result: The extraction result to evaluate rules against.

        Returns:
            List of triggered action dictionaries.
        """
        triggered_actions: List[Dict[str, Any]] = []

        for rule in self._rules:
            try:
                if self._check_condition(rule, extraction_result):
                    action = self._create_action(rule, extraction_result)
                    triggered_actions.append(action.to_dict())
                    logger.info(f"Rule triggered: {rule.get('name', 'unnamed')}")
            except Exception as e:
                logger.warning(
                    f"Error evaluating rule '{rule.get('name', 'unnamed')}': {e}"
                )

        return triggered_actions

    def _check_condition(
        self, rule: Dict[str, Any], result: ExtractionResult
    ) -> bool:
        """Check if a rule's condition is met by the extraction result.

        Supports operators: gt, lt, eq, contains, exists, and compound
        conditions with AND/OR.

        Args:
            rule: Rule definition with trigger conditions.
            result: Extraction result to check against.

        Returns:
            True if the condition is satisfied.
        """
        trigger = rule.get("trigger", {})
        condition_str = trigger.get("condition", "")

        if not condition_str:
            return False

        # Parse compound conditions (AND, OR)
        if " AND " in condition_str:
            parts = condition_str.split(" AND ")
            return all(
                self._evaluate_single_condition(part.strip(), result)
                for part in parts
            )
        elif " OR " in condition_str:
            parts = condition_str.split(" OR ")
            return any(
                self._evaluate_single_condition(part.strip(), result)
                for part in parts
            )
        else:
            return self._evaluate_single_condition(condition_str, result)

    def _evaluate_single_condition(
        self, condition: str, result: ExtractionResult
    ) -> bool:
        """Evaluate a single condition expression against extraction result.

        Supported forms:
        - "field > value" (gt)
        - "field < value" (lt)
        - "field == value" or "field = value" (eq)
        - "field contains value" (contains)
        - "field detected" or "field exists" (exists)
        - "NOT field" (negation of exists)

        Args:
            condition: Condition string to evaluate.
            result: Extraction result to check against.

        Returns:
            True if the condition is met.
        """
        entities = result.entities

        # Handle NOT prefix
        if condition.startswith("NOT "):
            inner = condition[4:].strip()
            return not self._evaluate_single_condition(inner, result)

        # Greater than: "field > value"
        gt_match = re.match(r"(\w+)\s*>\s*(.+)", condition)
        if gt_match:
            field = gt_match.group(1).strip()
            value_str = gt_match.group(2).strip()
            field_value = self._get_field_value(field, entities)
            if field_value is not None:
                try:
                    return float(field_value) > float(value_str)
                except (ValueError, TypeError):
                    return False
            return False

        # Less than: "field < value"
        lt_match = re.match(r"(\w+)\s*<\s*(.+)", condition)
        if lt_match:
            field = lt_match.group(1).strip()
            value_str = lt_match.group(2).strip()
            field_value = self._get_field_value(field, entities)
            if field_value is not None:
                try:
                    return float(field_value) < float(value_str)
                except (ValueError, TypeError):
                    return False
            return False

        # Equality: "field == value" or "field = value"
        eq_match = re.match(r"(\w+)\s*={1,2}\s*(.+)", condition)
        if eq_match:
            field = eq_match.group(1).strip()
            value_str = eq_match.group(2).strip()
            field_value = self._get_field_value(field, entities)
            if field_value is not None:
                return str(field_value).lower() == value_str.lower()
            return False

        # Contains: "field contains value"
        contains_match = re.match(r"(\w+)\s+contains\s+(.+)", condition, re.IGNORECASE)
        if contains_match:
            field = contains_match.group(1).strip()
            value_str = contains_match.group(2).strip()
            field_value = self._get_field_value(field, entities)
            if field_value is not None:
                if isinstance(field_value, list):
                    return any(value_str.lower() in str(item).lower() for item in field_value)
                return value_str.lower() in str(field_value).lower()
            return False

        # Exists/detected: "field detected" or "field exists"
        exists_match = re.match(r"(\w+)\s+(?:detected|exists)", condition, re.IGNORECASE)
        if exists_match:
            field = exists_match.group(1).strip()
            field_value = self._get_field_value(field, entities)
            return field_value is not None and field_value != "" and field_value != []

        # Simple field name check (exists)
        field_value = self._get_field_value(condition.strip(), entities)
        return field_value is not None and field_value != "" and field_value != []

    def _get_field_value(
        self, field: str, entities: Dict[str, Any]
    ) -> Any:
        """Get a field value from entities, supporting nested dot notation.

        Args:
            field: Field name, possibly with dot notation (e.g., "risk.score").
            entities: Entity dictionary to search in.

        Returns:
            Field value or None if not found.
        """
        # Try direct lookup first
        if field in entities:
            return entities[field]

        # Try with underscores replaced by dots and vice versa
        alt_field = field.replace("_", ".")
        if alt_field in entities:
            return entities[alt_field]

        alt_field = field.replace(".", "_")
        if alt_field in entities:
            return entities[alt_field]

        # Try nested dot notation
        parts = field.split(".")
        current = entities
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def _create_action(
        self, rule: Dict[str, Any], result: ExtractionResult
    ) -> Action:
        """Create an Action object from a triggered rule.

        Args:
            rule: The rule definition that was triggered.
            result: The extraction result that triggered the rule.

        Returns:
            Action instance ready for execution.
        """
        action_config = rule.get("action", {})
        rule_name = rule.get("name", "unnamed_rule")

        payload = {
            "document_id": str(result.document_id),
            "extraction_mode": result.extraction_mode.value,
            "rule_name": rule_name,
            "entities": result.entities,
            "overall_confidence": result.confidence_report.overall_confidence,
            "triggered_at": datetime.utcnow().isoformat(),
        }

        return Action(
            action_type=action_config.get("type", "webhook"),
            target=action_config.get("endpoint", ""),
            payload=payload,
            triggered_by_rule=rule_name,
            priority=action_config.get("priority", "medium"),
        )

    async def execute_action(self, action: Action) -> Dict[str, Any]:
        """Execute a triggered action by dispatching to the appropriate handler.

        Args:
            action: The action to execute.

        Returns:
            Execution result dictionary.
        """
        handlers = {
            "webhook": self._handle_webhook,
            "email": self._handle_email,
            "route": self._handle_route,
            "flag": self._handle_flag,
        }

        handler = handlers.get(action.action_type)
        if handler is None:
            logger.warning(f"Unknown action type: {action.action_type}")
            return {"status": "skipped", "reason": f"Unknown type: {action.action_type}"}

        return await handler(action)

    async def _handle_webhook(self, action: Action) -> Dict[str, Any]:
        """Handle webhook action dispatch.

        Args:
            action: Webhook action to execute.

        Returns:
            Execution result.
        """
        from src.utils.webhook_client import WebhookClient

        client = WebhookClient()
        try:
            results = await client.dispatch_event(
                event_type=f"action.{action.triggered_by_rule}",
                payload=action.payload,
            )
            return {"status": "dispatched", "results": results}
        except Exception as e:
            logger.error(f"Webhook dispatch failed: {e}")
            return {"status": "failed", "error": str(e)}

    async def _handle_email(self, action: Action) -> Dict[str, Any]:
        """Handle email action (placeholder for email integration).

        Args:
            action: Email action to execute.

        Returns:
            Execution result.
        """
        logger.info(f"Email action triggered: {action.target}")
        return {"status": "queued", "target": action.target}

    async def _handle_route(self, action: Action) -> Dict[str, Any]:
        """Handle document routing action.

        Args:
            action: Route action to execute.

        Returns:
            Execution result.
        """
        logger.info(f"Route action triggered: {action.target}")
        return {"status": "routed", "target": action.target}

    async def _handle_flag(self, action: Action) -> Dict[str, Any]:
        """Handle flag/alert action.

        Args:
            action: Flag action to execute.

        Returns:
            Execution result.
        """
        logger.info(
            f"Flag action triggered: {action.triggered_by_rule} -> {action.target}"
        )
        return {"status": "flagged", "rule": action.triggered_by_rule}
