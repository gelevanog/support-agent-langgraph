"""REST API and operator UI."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from support_agent.api import create_app
from support_agent.config import Settings
from tests.conftest import example


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def submit(client: TestClient, name: str) -> dict:  # type: ignore[type-arg]
    response = client.post("/tickets", json=example(name).ticket.model_dump(mode="json"))
    assert response.status_code == 201, response.text
    return response.json()  # type: ignore[no-any-return]


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["llm_provider"] == "fake"


def test_submit_and_get_ticket(client: TestClient) -> None:
    created = submit(client, "01")
    assert created["status"] == "resolved"
    assert created["resolution"] == "refund_issued"
    fetched = client.get(f"/tickets/{created['id']}").json()
    assert fetched["decision"]["verdict"] == "auto_approve"
    assert fetched["analysis"]["intent"] == "refund_request"
    assert [e["seq"] for e in fetched["audit"]] == list(range(len(fetched["audit"])))
    assert {e["kind"] for e in fetched["audit"]} >= {"llm", "tool", "rule", "decision", "action", "reply"}


def test_ticket_exposes_retrieved_articles(client: TestClient) -> None:
    created = submit(client, "06")
    knowledge = client.get(f"/tickets/{created['id']}").json()["knowledge"]
    assert knowledge["articles"][0] == {
        "id": "KB-001",
        "title": "International shipping",
        "score": knowledge["articles"][0]["score"],
        "snippet": knowledge["articles"][0]["snippet"],
        "in_reply_context": True,
        "cited": True,
    }
    page = client.get(f"/ui/tickets/{created['id']}").text
    assert "Knowledge base" in page
    assert "KB-001" in page
    assert "cited" in page


def test_submit_validates_input(client: TestClient) -> None:
    assert client.post("/tickets", json={"body": ""}).status_code == 422
    assert client.post("/tickets", json={"body": "hello there", "customer_email": "not-an-email"}).status_code == 422


def test_approve_flow(client: TestClient) -> None:
    paused = submit(client, "02")
    assert paused["status"] == "awaiting_approval"
    queue = client.get("/tickets", params={"status": "awaiting_approval"}).json()
    assert [t["id"] for t in queue] == [paused["id"]]
    assert queue[0]["summary"]

    approved = client.post(f"/tickets/{paused['id']}/approve", json={"operator": "maria", "note": "ok"})
    assert approved.status_code == 200
    assert approved.json()["resolution"] == "refund_issued"
    assert client.get("/tickets", params={"status": "awaiting_approval"}).json() == []
    # Reviewing again is a conflict.
    assert client.post(f"/tickets/{paused['id']}/approve").status_code == 409


def test_reject_flow_without_body(client: TestClient) -> None:
    paused = submit(client, "02")
    rejected = client.post(f"/tickets/{paused['id']}/reject")
    assert rejected.status_code == 200
    body = rejected.json()
    assert body["resolution"] == "rejected_after_review"
    assert any(e["message"].startswith("Rejected by operator") for e in body["audit"])


def test_unknown_ticket_is_404(client: TestClient) -> None:
    assert client.get("/tickets/TCK-NOPE").status_code == 404
    assert client.post("/tickets/TCK-NOPE/approve").status_code == 404


def test_list_rejects_unknown_status(client: TestClient) -> None:
    assert client.get("/tickets", params={"status": "bogus"}).status_code == 422


def test_store_api_is_mounted(client: TestClient) -> None:
    assert client.get("/store/orders/1042").json()["id"] == "1042"


# --- UI --------------------------------------------------------------------------------------


def test_ui_index_lists_pending_tickets_and_examples(client: TestClient) -> None:
    paused = submit(client, "02")
    page = client.get("/ui")
    assert page.status_code == 200
    assert paused["id"] in page.text
    assert "Awaiting approval" in page.text
    assert "Refund over $100 needs a human" in page.text  # example picker


def test_ui_ticket_detail_shows_rules_and_audit(client: TestClient) -> None:
    created = submit(client, "03")
    page = client.get(f"/ui/tickets/{created['id']}")
    assert page.status_code == 200
    assert "refund.within_window" in page.text
    assert "Audit trail" in page.text
    assert "outside the 30-day refund window" in page.text


def test_ui_submit_form_redirects_to_ticket(client: TestClient) -> None:
    response = client.post(
        "/ui/tickets",
        data={"body": "Where is my order #1045?", "customer_email": "emma.rossi@example.com", "channel": "chat"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"].startswith("/ui/tickets/TCK-")


def test_ui_htmx_approve_returns_fragment(client: TestClient) -> None:
    paused = submit(client, "02")
    response = client.post(
        f"/ui/tickets/{paused['id']}/approve",
        data={"operator": "maria", "note": ""},
        headers={"HX-Request": "true"},
    )
    assert response.status_code == 200
    assert "<html" not in response.text
    assert "Approved by maria" in response.text
    assert "refund issued" in response.text


def test_ui_reject_without_js_redirects(client: TestClient) -> None:
    paused = submit(client, "02")
    response = client.post(f"/ui/tickets/{paused['id']}/reject", data={"operator": "maria"}, follow_redirects=False)
    assert response.status_code == 303
    assert client.get(f"/tickets/{paused['id']}").json()["resolution"] == "rejected_after_review"


def test_root_redirects_to_ui(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)
    assert response.headers["location"] == "/ui"
