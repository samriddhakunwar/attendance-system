"""
Attendance analytics, written as SQL so the logic is easy to read and reuse
(the same queries work in DB Browser for SQLite, DBeaver, or any SQL tool).
Each function returns a pandas DataFrame for the dashboard.
"""
from __future__ import annotations

import pandas as pd

AT_RISK = 0.75   # common university rule: below 75% attendance = not allowed to sit exams


def _q(con, sql: str, params=()) -> pd.DataFrame:
    return pd.read_sql_query(sql, con, params=params)


def overview(con, class_name: str) -> dict:
    df = _q(con, """
        SELECT COUNT(DISTINCT se.id)                                   AS sessions,
               COUNT(DISTINCT a.student_id)                             AS students,
               AVG(a.status = 'present')                                AS attendance_rate,
               AVG(CASE WHEN a.status = 'present' THEN a.method = 'auto' END) AS auto_share
        FROM sessions se JOIN attendance a ON a.session_id = se.id
        WHERE se.class_name = ?""", (class_name,))
    return df.iloc[0].to_dict()


def student_summary(con, class_name: str) -> pd.DataFrame:
    """One row per student: sessions attended, rate, and current absence streak."""
    return _q(con, """
        WITH marks AS (
            SELECT st.id AS student_id, st.roll_no, st.name, se.session_date, se.id AS session_id,
                   a.status,
                   -- number each student's sessions from newest (1) to oldest
                   ROW_NUMBER() OVER (PARTITION BY st.id ORDER BY se.session_date DESC, se.id DESC) AS recency
            FROM attendance a
            JOIN sessions se ON se.id = a.session_id
            JOIN students st ON st.id = a.student_id
            WHERE se.class_name = ?
        ),
        latest_present AS (       -- most recent session the student attended
            SELECT student_id, MIN(recency) AS first_present_recency
            FROM marks WHERE status = 'present' GROUP BY student_id
        )
        SELECT m.roll_no, m.name,
               COUNT(*)                                  AS sessions,
               SUM(m.status = 'present')                 AS attended,
               ROUND(AVG(m.status = 'present'), 4)       AS rate,
               COALESCE(lp.first_present_recency - 1, COUNT(*)) AS current_absence_streak,
               MAX(CASE WHEN m.status = 'present' THEN m.session_date END) AS last_seen
        FROM marks m LEFT JOIN latest_present lp ON lp.student_id = m.student_id
        GROUP BY m.student_id
        ORDER BY rate, m.roll_no""", (class_name,))


def weekly_trend(con, class_name: str) -> pd.DataFrame:
    return _q(con, """
        -- weeks start on Sunday (the first teaching day where Saturday is the holiday)
        SELECT DATE(se.session_date, '-' || STRFTIME('%w', se.session_date) || ' days') AS week_start,
               COUNT(DISTINCT se.id)                          AS sessions,
               ROUND(AVG(a.status = 'present'), 4)            AS rate
        FROM sessions se JOIN attendance a ON a.session_id = se.id
        WHERE se.class_name = ?
        GROUP BY week_start ORDER BY week_start""", (class_name,))


def by_weekday(con, class_name: str) -> pd.DataFrame:
    df = _q(con, """
        SELECT CAST(STRFTIME('%w', se.session_date) AS INTEGER) AS dow,
               COUNT(DISTINCT se.id)                            AS sessions,
               ROUND(AVG(a.status = 'present'), 4)              AS rate
        FROM sessions se JOIN attendance a ON a.session_id = se.id
        WHERE se.class_name = ?
        GROUP BY dow ORDER BY dow""", (class_name,))
    df["day"] = df.dow.map({0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat"})
    return df


def session_list(con, class_name: str) -> pd.DataFrame:
    return _q(con, """
        SELECT se.id, se.session_date, se.subject,
               SUM(a.status = 'present')          AS present,
               COUNT(*)                           AS enrolled,
               ROUND(AVG(a.status = 'present'), 4) AS rate,
               se.faces_detected, se.unknown_faces,
               SUM(a.method = 'auto'      AND a.status = 'present') AS auto,
               SUM(a.method = 'confirmed')                          AS confirmed,
               SUM(a.method = 'manual')                             AS manual
        FROM sessions se JOIN attendance a ON a.session_id = se.id
        WHERE se.class_name = ?
        GROUP BY se.id ORDER BY se.session_date DESC, se.id DESC""", (class_name,))


def recognition_quality(con, class_name: str) -> pd.DataFrame:
    """How often did the teacher have to step in? A practical accuracy measure:
    'manual' corrections are the model's misses or mistakes."""
    return _q(con, """
        SELECT a.method, a.status, COUNT(*) AS marks
        FROM attendance a JOIN sessions se ON se.id = a.session_id
        WHERE se.class_name = ?
        GROUP BY a.method, a.status ORDER BY a.method, a.status""", (class_name,))


def register(con, class_name: str) -> pd.DataFrame:
    """Classic attendance register: students down, dates across, P / A in the cells."""
    long = _q(con, """
        SELECT st.roll_no, st.name,
               se.session_date || CASE WHEN se.subject <> '' THEN ' ' || se.subject ELSE '' END AS session,
               CASE a.status WHEN 'present' THEN 'P' ELSE 'A' END AS mark
        FROM attendance a
        JOIN sessions se ON se.id = a.session_id
        JOIN students st ON st.id = a.student_id
        WHERE se.class_name = ?""", (class_name,))
    if long.empty:
        return long
    wide = long.pivot_table(index=["roll_no", "name"], columns="session", values="mark",
                            aggfunc="first").reset_index()
    marks = wide.columns[2:]
    wide["attended"] = (wide[marks] == "P").sum(axis=1)
    wide["rate"] = (wide["attended"] / wide[marks].notna().sum(axis=1)).round(3)
    return wide
