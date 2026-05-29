"""Webhook management endpoints."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.utils.webhook_client import WebhookClient

router = APIRouter()

# Shared webhook client instance
_webhook_client = WebhookClient()


class WebhookCreate(BaseModel):
    """Request model for registering a webhook."""

    url: str = Field(..., description="Webhook endpoint URL")
    events: List[str] = Field(..., description="Event types to subscribe to")
    secret: Optional[str] = Field(None, description="Optional webhook secret for signing")


@router.get("")
async def list_webhooks():
    """List all registered webhooks."""
    webhooks = _webhook_client.list_webhooks()
    return {"webhooks": webhooks}


@router.post("")
async def register_webhook(webhook: WebhookCreate):
    """Register a new webhook endpoint."""
    webhook_id = _webhook_client.register_webhook(
        url=webhook.url,
        events=webhook.events,
        secret=webhook.secret,
    )
    return {
        "id": webhook_id,
        "url": webhook.url,
        "events": webhook.events,
        "status": "registered",
    }


@router.delete("/{webhook_id}")
async def remove_webhook(webhook_id: str):
    """Remove a registered webhook."""
    removed = _webhook_client.remove_webhook(webhook_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Webhook not found: {webhook_id}")
    return {"id": webhook_id, "status": "removed"}


@router.post("/{webhook_id}/test")
async def test_webhook(webhook_id: str):
    """Send a test event to a specific webhook."""
    # Find the webhook
    webhooks = _webhook_client.list_webhooks()
    target = None
    for wh in webhooks:
        if wh["id"] == webhook_id:
            target = wh
            break

    if target is None:
        raise HTTPException(status_code=404, detail=f"Webhook not found: {webhook_id}")

    # Send test event
    test_payload = {
        "test": True,
        "message": "This is a test webhook delivery",
        "webhook_id": webhook_id,
    }

    try:
        results = await _webhook_client.dispatch_event(
            event_type="test",
            payload=test_payload,
        )
        return {
            "webhook_id": webhook_id,
            "status": "test_sent",
            "results": results,
        }
    except Exception as e:
        return {
            "webhook_id": webhook_id,
            "status": "test_failed",
            "error": str(e),
        }
