"""SQLite database connection management for LCM.

Handles connection setup, WAL mode configuration, and connection pooling.
"""

import sqlite3
import threading
from contextlib import contextmanager
from typing import Optional, Generator
from .config import LcmConfig


class LcmDatabase:
    """SQLite database connection manager for LCM.
    
    Manages a single database connection with proper WAL mode setup
    and thread safety for the LCM context engine.
    """
    
    def __init__(self, config: LcmConfig):
        self.config = config
        self.database_path = config.database_path
        self._connection: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with proper configuration."""
        conn = sqlite3.connect(
            self.database_path,
            check_same_thread=False,  # We handle thread safety manually
            timeout=30.0,  # 30 second lock timeout
            isolation_level=None  # Autocommit mode, we'll handle transactions manually
        )
        
        # Configure SQLite settings for performance and reliability
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL") 
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB
        
        # Set reasonable timeouts
        conn.execute("PRAGMA busy_timeout=30000")  # 30 second busy timeout
        
        return conn
    
    @property 
    def connection(self) -> sqlite3.Connection:
        """Get the database connection, creating it if needed."""
        if self._connection is None:
            with self._lock:
                if self._connection is None:
                    self._connection = self._create_connection()
        return self._connection
    
    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database transactions with automatic rollback on error."""
        conn = self.connection
        with self._lock:
            try:
                conn.execute("BEGIN")
                yield conn
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass  # Rollback failed, but preserve the original exception
                raise
    
    def execute(self, sql: str, params=None) -> sqlite3.Cursor:
        """Execute a single SQL statement."""
        conn = self.connection
        with self._lock:
            if params is None:
                return conn.execute(sql)
            else:
                return conn.execute(sql, params)
    
    def executemany(self, sql: str, params_list) -> sqlite3.Cursor:
        """Execute a SQL statement with multiple parameter sets."""
        conn = self.connection
        with self._lock:
            return conn.executemany(sql, params_list)
    
    def close(self):
        """Close the database connection."""
        if self._connection is not None:
            with self._lock:
                if self._connection is not None:
                    self._connection.close()
                    self._connection = None
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup


# Global database instance (initialized by context engine)
_db_instance: Optional[LcmDatabase] = None


def get_database() -> Optional[LcmDatabase]:
    """Get the global database instance."""
    return _db_instance


def initialize_database(config: LcmConfig) -> LcmDatabase:
    """Initialize the global database instance."""
    global _db_instance
    if _db_instance is not None:
        _db_instance.close()
    _db_instance = LcmDatabase(config)
    return _db_instance


def close_database():
    """Close the global database instance."""
    global _db_instance
    if _db_instance is not None:
        _db_instance.close()
        _db_instance = None