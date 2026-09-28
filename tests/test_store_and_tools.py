"""Mock Store API, HTTP client and LangChain tool wrappers."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from support_agent.store_api import StoreRepository, create_store_app
from support_agent.tools import StoreClient, StoreConflictError, StoreNotFoundError, build_store_tools

TODAY = date(2026, 9, 28)


@pytest.fixture
def repo() -> StoreRepository:
    return StoreRepository(today=TODAY)


@pytest.fixture
def http(repo: StoreRepository) -> TestClient:
    return TestClient(create_store_app(repo))


# --- Store API -------------------------------------------------------------------------------


def test_seed_dates_are_relative_to_reference_date(repo: StoreRepository) -> None:
    order = repo.get_order("1044")
    assert order.delivered_on == TODAY - timedelta(days=45)


def test_get_order(http: TestClient) -> None:
    response = http.get("/orders/1042")
    assert response.status_code == 200
    body = response.json()
    assert body["customer_email"] == "anna.miller@example.com"
    assert body["total"] == "64.90"


def test_get_order_not_found(http: TestClient) -> None:
    response = http.get("/orders/9999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_get_customer_is_case_insensitive(http: TestClient) -> None:
    response = http.get("/customers/Chloe.Nguyen@Example.com")
    assert response.status_code == 200
    assert response.json()["tier"] == "vip"


@pytest.mark.parametrize(
    ("query", "expected_top"),
    [
        ("Do you ship to Canada? How long does international shipping take?", "KB-001"),
        ("how long until my refund shows up on my card", "KB-004"),
        ("warranty on my espresso machine", "KB-005"),
        ("how do I clean the burr grinder", "KB-008"),
    ],
)
def test_knowledge_base_search_ranks_relevant_article_first(http: TestClient, query: str, expected_top: str) -> None:
    response = http.get("/kb/search", params={"q": query})
    assert response.status_code == 200
    assert response.json()[0]["id"] == expected_top


def test_knowledge_base_search_without_match_returns_empty(http: TestClient) -> None:
    assert http.get("/kb/search", params={"q": "zebra"}).json() == []


def test_create_refund_updates_order(http: TestClient, repo: StoreRepository) -> None:
    response = http.post("/orders/1042/refunds", json={"amount": "64.90", "reason": "cracked"})
    assert response.status_code == 201
    assert response.json()["id"].startswith("RF-")
    assert repo.get_order("1042").refunded_amount == Decimal("64.90")
    # A second refund would exceed the refundable balance.
    assert http.post("/orders/1042/refunds", json={"amount": "1", "reason": "again"}).status_code == 409


def test_refund_of_undelivered_order_is_rejected(http: TestClient) -> None:
    response = http.post("/orders/1045/refunds", json={"amount": "10", "reason": "x"})
    assert response.status_code == 409


def test_update_shipping_address(http: TestClient) -> None:
    ok = http.put("/orders/1047/shipping-address", json={"shipping_address": "12 Beacon Street, Boston"})
    assert ok.status_code == 200
    assert ok.json()["shipping_address"] == "12 Beacon Street, Boston"
    shipped = http.put("/orders/1046/shipping-address", json={"shipping_address": "55 Lake Shore Drive"})
    assert shipped.status_code == 409


def test_create_escalation(http: TestClient, repo: StoreRepository) -> None:
    response = http.post("/escalations", json={"ticket_id": "TCK-1", "reason": "angry", "priority": "high"})
    assert response.status_code == 201
    assert repo.escalations[0].priority == "high"


# --- Client and tools ------------------------------------------------------------------------


async def test_client_maps_http_errors(repo: StoreRepository) -> None:
    async with StoreClient.in_process(create_store_app(repo)) as client:
        order = await client.get_order("#1042")
        assert order.id == "1042"
        with pytest.raises(StoreNotFoundError):
            await client.get_order("9999")
        with pytest.raises(StoreConflictError):
            await client.update_shipping_address("1046", "55 Lake Shore Drive")


async def test_read_only_tool_set() -> None:
    async with StoreClient.in_process() as client:
        tools = build_store_tools(client)
        assert [t.name for t in tools.read_only] == ["get_order", "get_customer", "search_knowledge_base"]


async def test_tool_returns_content_and_artifact(repo: StoreRepository) -> None:
    async with StoreClient.in_process(create_store_app(repo)) as client:
        tools = build_store_tools(client)
        call = {"name": "get_order", "args": {"order_id": "1043"}, "id": "c1", "type": "tool_call"}
        message = await tools.get_order.ainvoke(call)
        assert message.name == "get_order"
        assert message.artifact["order"]["total"] == "249.00"
        assert '"id": "1043"' in message.content


async def test_tool_errors_are_returned_not_raised(repo: StoreRepository) -> None:
    async with StoreClient.in_process(create_store_app(repo)) as client:
        tools = build_store_tools(client)
        call = {"name": "get_order", "args": {"order_id": "9999"}, "id": "c1", "type": "tool_call"}
        message = await tools.get_order.ainvoke(call)
        assert message.artifact == {"error": "Order 9999 not found", "status_code": 404}


async def test_write_tools_hit_the_store(repo: StoreRepository) -> None:
    async with StoreClient.in_process(create_store_app(repo)) as client:
        tools = build_store_tools(client)
        refund = await tools.create_refund.ainvoke(
            {
                "name": "create_refund",
                "args": {"order_id": "1042", "amount": "10.00", "reason": "partial"},
                "id": "c1",
                "type": "tool_call",
            }
        )
        assert refund.artifact["refund"]["amount"] == "10.00"
        escalation = await tools.escalate_to_human.ainvoke(
            {
                "name": "escalate_to_human",
                "args": {"ticket_id": "T-1", "reason": "r", "priority": "high"},
                "id": "c2",
                "type": "tool_call",
            }
        )
        assert escalation.artifact["escalation"]["queue"] == "tier-2"
        assert len(repo.refunds) == 1
        assert len(repo.escalations) == 1
