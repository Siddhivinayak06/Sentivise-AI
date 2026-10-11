"""Thread-safe SQLite Database Connection Manager."""
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from config import DATABASE_PATH

# Local thread storage for database connections
_local = threading.local()

def get_connection(db_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    """Returns a thread-local SQLite connection with proper PRAGMAs."""
    if not hasattr(_local, "connection") or _local.connection is None:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high concurrent read performance and foreign keys
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        _local.connection = conn
    return _local.connection

@contextmanager
def get_db(db_path: Path = DATABASE_PATH):
    """Context manager for safe transactional database queries."""
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
