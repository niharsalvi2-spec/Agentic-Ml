"""
Database and Persistence Client (Supabase / Postgres / SQLite).
"""

import os
from typing import Optional

try:
    from supabase import create_client, Client
except ImportError:
    create_client = None
    Client = None

class DatabaseManager:
    """
    Central database client for user profiles, subscriptions, and experiment metadata.
    """
    def __init__(self):
        self.url: str = os.getenv("SUPABASE_URL", "")
        self.key: str = os.getenv("SUPABASE_KEY", "")
        self.client: Optional[Any] = None
        
        if create_client and self.url and self.key:
            try:
                self.client = create_client(self.url, self.key)
            except Exception:
                self.client = None

    def is_connected(self) -> bool:
        return self.client is not None

db = DatabaseManager()
