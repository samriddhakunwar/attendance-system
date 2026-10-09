"""
SQLite storage. One file (data/attendance.db) holds everything:

  students      roll number, name, class
  face_refs     the face "embeddings" used to recognise each student, plus a small
                face thumbnail for the roster. Full enrollment photos are NOT kept.
  sessions      one row per class meeting (date, class, subject, how many faces were seen)
  attendance    one row per student per session: present/absent, and HOW it was decided
                (auto = matched by the model, confirmed = teacher accepted a suggestion,
                 manual = teacher changed it by hand)
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "attendance.db"
DEMO_DB = ROOT / "data" / "demo.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    id          INTEGER PRIMARY KEY,
    roll_no     TEXT NOT NULL,
    name        TEXT NOT NULL,
    class_name  TEXT NOT NULL,
    active      INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (class_name, roll_no)
);
CREATE TABLE IF NOT EXISTS face_refs (
    id          INTEGER PRIMARY KEY,
    student_id  INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    embedding   BLOB NOT NULL,          -- 128 float32 values
    thumbnail   BLOB,                   -- small JPEG of the face
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY,
    class_name      TEXT NOT NULL,
    session_date    TEXT NOT NULL,      -- YYYY-MM-DD
    subject         TEXT NOT NULL DEFAULT '',
    faces_detected  INTEGER NOT NULL DEFAULT 0,
    unknown_faces   INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (class_name, session_date, subject)
);
CREATE TABLE IF NOT EXISTS attendance (
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    student_id  INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    status      TEXT NOT NULL CHECK (status IN ('present', 'absent')),
    method      TEXT NOT NULL CHECK (method IN ('auto', 'confirmed', 'manual')),
    similarity  REAL,
    PRIMARY KEY (session_id, student_id)
);
CREATE INDEX IF NOT EXISTS idx_att_student ON attendance(student_id);
"""


@contextmanager
def connect(path: Path = DEFAULT_DB):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    try:
        yield con
        con.commit()
    finally:
        con.close()


# ------------------------------------------------------------------ students
def add_student(con, roll_no: str, name: str, class_name: str) -> int:
    cur = con.execute("INSERT INTO students (roll_no, name, class_name) VALUES (?, ?, ?)",
                      (roll_no.strip(), name.strip(), class_name.strip()))
    return cur.lastrowid


def add_face_ref(con, student_id: int, embedding: np.ndarray, thumbnail_jpg: bytes | None) -> None:
    con.execute("INSERT INTO face_refs (student_id, embedding, thumbnail) VALUES (?, ?, ?)",
                (student_id, embedding.astype(np.float32).tobytes(), thumbnail_jpg))


def delete_student(con, student_id: int) -> None:
    """Removes the student, their face data and their attendance rows."""
    con.execute("DELETE FROM students WHERE id = ?", (student_id,))


def classes(con) -> list[str]:
    return [r[0] for r in con.execute(
        "SELECT DISTINCT class_name FROM students WHERE active = 1 ORDER BY class_name")]


def roster(con, class_name: str) -> list[sqlite3.Row]:
    return con.execute("""
        SELECT s.id, s.roll_no, s.name, COUNT(f.id) AS n_photos,
               (SELECT thumbnail FROM face_refs WHERE student_id = s.id ORDER BY id LIMIT 1) AS thumb
        FROM students s LEFT JOIN face_refs f ON f.student_id = s.id
        WHERE s.class_name = ? AND s.active = 1
        GROUP BY s.id ORDER BY s.roll_no""", (class_name,)).fetchall()


def gallery(con, class_name: str) -> dict[int, np.ndarray]:
    """student_id -> (k, 128) array of reference embeddings, for one class."""
    out: dict[int, list[np.ndarray]] = {}
    for sid, blob in con.execute("""
            SELECT f.student_id, f.embedding FROM face_refs f
            JOIN students s ON s.id = f.student_id
            WHERE s.class_name = ? AND s.active = 1""", (class_name,)):
        out.setdefault(sid, []).append(np.frombuffer(blob, np.float32))
    return {k: np.stack(v) for k, v in out.items()}


# ------------------------------------------------------------------ sessions
def save_session(con, class_name: str, session_date: date, subject: str,
                 marks: dict[int, tuple[str, str, float | None]],
                 faces_detected: int, unknown_faces: int) -> int:
    """marks: student_id -> (status, method, similarity). Re-saving the same class,
    date and subject replaces the earlier record."""
    con.execute("DELETE FROM sessions WHERE class_name = ? AND session_date = ? AND subject = ?",
                (class_name, session_date.isoformat(), subject))
    cur = con.execute("""INSERT INTO sessions (class_name, session_date, subject, faces_detected, unknown_faces)
                         VALUES (?, ?, ?, ?, ?)""",
                      (class_name, session_date.isoformat(), subject, faces_detected, unknown_faces))
    sid = cur.lastrowid
    con.executemany("INSERT INTO attendance VALUES (?, ?, ?, ?, ?)",
                    [(sid, st, status, method, sim) for st, (status, method, sim) in marks.items()])
    return sid


def delete_session(con, session_id: int) -> None:
    con.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
