"""Webhook client for dispatching events to registered endpoints."""

import hashlib
import hmac
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx


class WebhookRegistration:
    """Represents a registered webhook endpoint."""

    def __init__(
        self, url: str, events: List[str], secret: Optional[str] = None
    ) -> None:
        self.id = str(uuid.uuid4())
        self.url = url
        self.events = events
        self.secret = secret
        self.created_at = datetime.utcnow()
        self.active = True


class WebhookClient:
    """Client for managing and dispatching webhook events."""

    def __init__(self, default_secret: str = "") -> None:
        """Initialize the webhook client.

        Args:
            default_secret: Default secret for HMAC signing.
        """
        self.default_secret = default_secret
        self._webhooks: Dict[str, WebhookRegistration] = {}

    def register_webhook(
        self, url: str, events: List[str], secret: Optional[str] = None
    ) -> str:
        """Register a new webhook endpoint.

        Args:
            url: Webhook URL to call.
            events: List of event types to subscribe to.
            secret: Optional per-webhook secret for signing.

        Returns:
            Webhook registration ID.
        """
        registration = WebhookRegistration(
            url=url, events=events, secret=secret or self.default_secret
        )
        self._webhooks[registration.id] = registration
        return registration.id

    def remove_webhook(self, webhook_id: str) -> bool:
        """Remove a registered webhook.

        Args:
            webhook_id: ID of the webhook to remove.

        Returns:
            True if the webhook was found and removed.
        """
        if webhook_id in self._webhooks:
            del self._webhooks[webhook_id]
            return True
        return False

    def list_webhooks(self) -> List[Dict[str, Any]]:
        """List all registered webhooks.

        Returns:
            List of webhook registration details.
        """
        return [
            {
                "id": wh.id,
                "url": wh.url,
                "events": wh.events,
                "active": wh.active,
                "created_at": wh.created_at.isoformat(),
            }
            for wh in self._webhooks.values()
        ]

    def _sign_payload(self, payload: str, secret: str) -> str:
        """Generate HMAC-SHA256 signature for a payload.

        Args:
            payload: JSON payload string.
            secret: Secret key for signing.

        Returns:
            Hex-encoded HMAC signature.
        """
        return hmac.new(
            key=secret.encode("utf-8"),
            msg=payload.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()

    async def dispatch_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        max_retries: int = 3,
    ) -> List[Dict[str, Any]]:
        """Dispatch an event to all registered webhooks that subscribe to it.

        Args:
            event_type: Type of event to dispatch.
            payload: Event payload data.
            max_retries: Maximum number of retry attempts.

        Returns:
            List of dispatch results (success/failure per webhook).
        """
        results = []
        matching_webhooks = [
            wh
            for wh in self._webhooks.values()
            if wh.active and event_type in wh.events
        ]

        for webhook in matching_webhooks:
            result = await self._send_webhook(webhook, event_type, payload, max_retries)
            results.append(result)

        return results

    async def _send_webhook(
        self,
        webhook: WebhookRegistration,
        event_type: str,
        payload: Dict[str, Any],
        max_retries: int,
    ) -> Dict[str, Any]:
        """Send a webhook request with retry logic.

        Args:
            webhook: Webhook registration to send to.
            event_type: Event type being dispatched.
            payload: Event payload.
            max_retries: Maximum retries on failure.

        Returns:
            Result dictionary with status and details.
        """
        body = json.dumps(
            {
                "event_type": event_type,
                "payload": payload,
                "timestamp": datetime.utcnow().isoformat(),
                "webhook_id": webhook.id,
            }
        )

        headers = {
            "Content-Type": "application/json",
            "X-Event-Type": event_type,
        }

        if webhook.secret:
            signature = self._sign_payload(body, webhook.secret)
            headers["X-Webhook-Signature"] = f"sha256={signature}"

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        webhook.url, content=body, headers=headers
                    )
                    if response.status_code < 400:
                        return {
                            "webhook_id": webhook.id,
                            "status": "success",
                            "status_code": response.status_code,
                            "attempt": attempt + 1,
                        }
            except Exception as e:
                if attempt == max_retries - 1:
                    return {
                        "webhook_id": webhook.id,
                        "status": "failed",
                        "error": str(e),
                        "attempts": max_retries,
                    }

        return {
            "webhook_id": webhook.id,
            "status": "failed",
            "error": "Max retries exceeded",
            "attempts": max_retries,
        }
