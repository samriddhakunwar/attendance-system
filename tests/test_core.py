"""Quick checks that need no photos or models. Run:  python -m pytest -q"""
from datetime import date

import numpy as np

from attendance import analytics as A
from attendance import db
from attendance.engine import Face, _reading_order, match


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def face(vec, x=0, y=0):
    return Face(box=(x, y, 50, 50), score=0.9, embedding=unit(vec))


def test_match_is_one_to_one_and_respects_threshold():
    gallery = {1: unit([1, 0, 0])[None], 2: unit([0, 1, 0])[None]}
    faces = [face([1, 0.1, 0]), face([1, 0.2, 0]), face([0, 0, 1])]   # two look like #1, one stranger
    out = match(faces, gallery, threshold=0.4)
    ids = [f.student_id for f in out]
    assert ids.count(1) == 1                 # student 1 claimed once, by the closest face
    assert out[0].student_id == 1
    assert out[2].student_id is None         # the stranger stays unknown
    assert 2 not in ids                      # nobody resembles student 2


def test_reading_order_rows_then_left_to_right():
    faces = [face([1, 0, 0], x=300, y=10), face([1, 0, 0], x=10, y=20),
             face([1, 0, 0], x=150, y=140), face([1, 0, 0], x=20, y=150)]
    xs = [(f.box[0], f.box[1]) for f in _reading_order(faces)]
    assert xs == [(10, 20), (300, 10), (20, 150), (150, 140)]


def test_session_save_and_analytics(tmp_path):
    path = tmp_path / "t.db"
    with db.connect(path) as con:
        a = db.add_student(con, "01", "A", "C")
        b = db.add_student(con, "02", "B", "C")
        for d, (sa, sb) in enumerate([("present", "present"), ("present", "absent"),
                                      ("absent", "absent"), ("absent", "absent")], 1):
            db.save_session(con, "C", date(2026, 8, d), "", {a: (sa, "auto", None), b: (sb, "auto", None)}, 2, 0)
        # saving the same class/date again replaces it instead of duplicating
        db.save_session(con, "C", date(2026, 8, 4), "", {a: ("present", "manual", None), b: ("absent", "auto", None)}, 1, 0)
        s = A.student_summary(con, "C").set_index("roll_no")
        assert s.loc["01", "sessions"] == 4 and s.loc["01", "attended"] == 3
        assert s.loc["01", "current_absence_streak"] == 0
        assert s.loc["02", "current_absence_streak"] == 3     # absent in the last 3 classes
        assert A.overview(con, "C")["sessions"] == 4
        reg = A.register(con, "C")
        assert list(reg.loc[reg.roll_no == "02", "attended"]) == [1]
        db.delete_student(con, b)                              # cascades to attendance + faces
        assert con.execute("SELECT COUNT(*) FROM attendance WHERE student_id = ?", (b,)).fetchone()[0] == 0
