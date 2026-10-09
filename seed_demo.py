"""
Creates data/demo.db: a made-up class with 10 weeks of attendance, so the analytics
dashboard can be explored (and screenshotted) without real students.

Everything here is synthetic: invented names, no faces, randomly generated marks with
a few realistic patterns built in (a handful of students with poor attendance, one
student who stops coming mid-semester, lower turnout on Fridays). Classes run Sunday
to Friday, with Saturday off, as in Nepal.

Run:  python seed_demo.py
"""
import random
from datetime import date, timedelta

from attendance import db

SEED = 42
CLASS = "BSc CSIT - Semester 3 (demo)"
SUBJECT = "Database Management Systems"
START, END = date(2026, 7, 26), date(2026, 10, 2)   # a 10-week stretch, Sun..Fri

FIRST = ["Aarav", "Aashika", "Abhishek", "Anisha", "Bibek", "Bipana", "Dipesh", "Ishita",
         "Kiran", "Kritika", "Manish", "Nisha", "Pratik", "Prakriti", "Rohan", "Sabina",
         "Sagar", "Samjhana", "Sandesh", "Shreya", "Sujan", "Sunita", "Suraj", "Trishna",
         "Utsav", "Yubraj", "Anjali", "Bishal", "Dikshya", "Nabin", "Pooja", "Rabin",
         "Rojina", "Sameer", "Srijana", "Ujjwal"]
LAST = ["Adhikari", "Bhandari", "Gurung", "Karki", "KC", "Magar", "Maharjan", "Pandey",
        "Poudel", "Rai", "Sharma", "Shrestha", "Tamang", "Thapa", "Shakya", "Khadka"]


def main() -> None:
    rng = random.Random(SEED)
    if db.DEMO_DB.exists():
        db.DEMO_DB.unlink()
    with db.connect(db.DEMO_DB) as con:
        students = []
        for k, first in enumerate(FIRST, 1):
            sid = db.add_student(con, f"{k:02d}", f"{first} {rng.choice(LAST)}", CLASS)
            p = min(0.98, max(0.55, rng.gauss(0.88, 0.06)))      # personal attendance habit
            students.append([sid, p])
        for i in rng.sample(range(len(students)), 4):            # four chronic absentees
            students[i][1] = rng.uniform(0.55, 0.68)
        dropout = rng.randrange(len(students))                   # stops attending in week 8

        days = [START + timedelta(d) for d in range((END - START).days + 1)]
        days = [d for d in days if d.weekday() != 5]             # Saturday off
        for n, day in enumerate(days):
            day_effect = {4: -0.07, 6: -0.03}.get(day.weekday(), 0.0)   # Fri, Sun dips
            exam_week = 0.05 if 49 <= n < 55 else 0.0                    # turnout up before exams
            marks, present_n = {}, 0
            for i, (sid, p) in enumerate(students):
                prob = p + day_effect + exam_week
                if i == dropout and n >= 42:
                    prob = 0.03
                present = rng.random() < prob
                present_n += present
                if present:
                    r = rng.random()
                    method = "auto" if r < 0.93 else "confirmed" if r < 0.985 else "manual"
                    sim = round(rng.uniform(0.45, 0.85), 3) if method == "auto" else \
                        round(rng.uniform(0.30, 0.40), 3) if method == "confirmed" else None
                    marks[sid] = ("present", method, sim)
                else:
                    marks[sid] = ("absent", "manual" if rng.random() < 0.01 else "auto", None)
            unknown = rng.choice([0, 0, 0, 1, 1, 2])
            db.save_session(con, CLASS, day, SUBJECT, marks, present_n + unknown, unknown)
    print(f"Wrote {db.DEMO_DB}: {len(students)} students, {len(days)} sessions")


if __name__ == "__main__":
    main()
