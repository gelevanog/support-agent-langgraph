"""Mock Store API: the fictional shop's orders, customers, knowledge base and helpdesk."""

from support_agent.store_api.app import create_store_app
from support_agent.store_api.repository import StoreRepository

__all__ = ["StoreRepository", "create_store_app"]
