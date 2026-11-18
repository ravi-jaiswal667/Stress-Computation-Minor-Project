
import sqlite3
import json
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent
DB_PATH = ROOT_DIR / "sessions.db"


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                status TEXT NOT NULL DEFAULT 'active'
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                emotion TEXT NOT NULL,
                confidence REAL NOT NULL,
                experimental_score REAL NOT NULL,
                score_category TEXT NOT NULL,
                emotion_scores TEXT NOT NULL,
                FOREIGN KEY (session_id)
                    REFERENCES sessions(session_id)
            )
        """)

        # Check whether instantaneous_score already exists
        columns = [
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(readings)"
            ).fetchall()
        ]

        if "instantaneous_score" not in columns:

            connection.execute("""
                ALTER TABLE readings
                ADD COLUMN instantaneous_score REAL
            """)

            # Preserve old records using their previously stored score
            connection.execute("""
                UPDATE readings
                SET instantaneous_score = experimental_score
                WHERE instantaneous_score IS NULL
            """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_readings_session
            ON readings(session_id)
        """)


def create_session(session_id):
    started_at = datetime.now().isoformat(timespec="seconds")

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO sessions
            (session_id, started_at, status)
            VALUES (?, ?, 'active')
            """,
            (session_id, started_at)
        )

    return started_at


def save_reading(session_id, reading):
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO readings (
                session_id,
                timestamp,
                emotion,
                confidence,
                instantaneous_score,
                experimental_score,
                score_category,
                emotion_scores
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                reading["timestamp"],
                reading["emotion"],
                reading["confidence"],
                reading.get(
                    "instantaneous_score",
                    reading["experimental_score"]
                ),
                reading["experimental_score"],
                reading["score_category"],
                json.dumps(reading["emotion_scores"])
            )
        )


def get_session(session_id):
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT * FROM sessions
            WHERE session_id = ?
            """,
            (session_id,)
        ).fetchone()

    return dict(row) if row else None


def get_readings(session_id):
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM readings
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,)
        ).fetchall()

    readings = []

    for row in rows:
        reading = dict(row)

        reading["emotion_scores"] = json.loads(
            reading["emotion_scores"]
        )

        # Fallback for any historical record
        if reading["instantaneous_score"] is None:
            reading["instantaneous_score"] = (
                reading["experimental_score"]
            )

        del reading["id"]
        del reading["session_id"]

        readings.append(reading)

    return readings


def finish_session(session_id):
    ended_at = datetime.now().isoformat(timespec="seconds")

    with get_connection() as connection:
        connection.execute(
            """
            UPDATE sessions
            SET ended_at = ?, status = 'completed'
            WHERE session_id = ?
            """,
            (ended_at, session_id)
        )

    return ended_at


def get_all_sessions():
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                s.session_id,
                s.started_at,
                s.ended_at,
                s.status,
                COUNT(r.id) AS total_readings
            FROM sessions s
            LEFT JOIN readings r
                ON s.session_id = r.session_id
            GROUP BY s.session_id
            ORDER BY s.started_at DESC
            """
        ).fetchall()

    return [dict(row) for row in rows]


def delete_session(session_id):
    with get_connection() as connection:
        connection.execute(
            "DELETE FROM readings WHERE session_id = ?",
            (session_id,)
        )

        connection.execute(
            "DELETE FROM sessions WHERE session_id = ?",
            (session_id,)
        )
