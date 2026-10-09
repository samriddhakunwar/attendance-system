"""
Class Attendance - mark attendance from a photo of the class.

Run:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import io
import json
import sqlite3
from datetime import date
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from attendance import analytics as A
from attendance import db
from attendance.engine import (DEFAULT_THRESHOLD, REVIEW_FLOOR, FaceEngine, annotate,
                               face_thumbnail, load_image, match)

ROOT = Path(__file__).parent
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9aa0aa"
NOT_A_STUDENT = "Not a student"

st.set_page_config(page_title="Class Attendance", page_icon=":material/how_to_reg:", layout="wide")


@st.cache_resource(show_spinner="Loading face models (first run downloads about 38 MB)...")
def get_engine() -> FaceEngine:
    return FaceEngine()


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### Class Attendance")
    which = st.radio("Data", ["My classes", "Demo class (made-up data)"], key="which_db",
                     help="The demo class has 10 weeks of invented attendance so you can "
                          "explore the dashboard. It has no faces, so photo attendance "
                          "only works with your own classes.")
    DB_PATH = db.DEFAULT_DB if which == "My classes" else db.DEMO_DB
    if DB_PATH == db.DEMO_DB and not DB_PATH.exists():
        import seed_demo
        seed_demo.main()
    threshold = st.slider("Match strictness", 0.30, 0.60, DEFAULT_THRESHOLD, 0.01,
                          help="How similar a face must be to a student's enrollment photo to "
                               f"count as a match. {DEFAULT_THRESHOLD} was chosen from the accuracy "
                               "test. Higher = fewer wrong matches but more faces to confirm.")
    st.caption("Face data never leaves this computer. Only face 'fingerprints' (numbers) "
               "and small thumbnails are stored; class photos are not saved.")


def conn():
    return db.connect(DB_PATH)


def class_picker(label="Class", key="class") -> str | None:
    with conn() as con:
        options = db.classes(con)
    if not options:
        return None
    return st.selectbox(label, options, key=key)


# ================================================================== pages
def page_take_attendance():
    st.title("Take attendance")
    with conn() as con:
        has_classes = bool(db.classes(con))
    if not has_classes:
        st.info("No students yet. Go to **Students** to enroll your class first.")
        return
    c1, c2, c3 = st.columns([2, 1, 2])
    with c1:
        cls = class_picker()
    with c2:
        day = st.date_input("Date", value=date.today(), format="YYYY-MM-DD")
    with c3:
        subject = st.text_input("Subject or period (optional)", placeholder="e.g. DBMS")

    with conn() as con:
        gal = db.gallery(con, cls)
        roster = db.roster(con, cls)
    if not gal:
        st.warning("Nobody in this class has an enrollment photo yet. Add photos under **Students**.")
        return
    names = {r["id"]: f"{r['roll_no']} {r['name']}" for r in roster}

    files = st.file_uploader("Class photo(s)", type=["jpg", "jpeg", "png", "webp"],
                             accept_multiple_files=True,
                             help="For a big room, upload 2-3 photos from different spots. "
                                  "A student matched in any photo is marked present.")
    if not files:
        st.caption("Tip: take the photo from the front with everyone facing the camera. "
                   "Faces should be at least ~40 pixels tall; phone photos work well.")
        return

    key = hashlib.md5(b"".join(f.getvalue() for f in files) + cls.encode()
                      + str(threshold).encode() + str(DB_PATH).encode()).hexdigest()
    if st.session_state.get("result_key") != key:
        engine = get_engine()
        results = []
        with st.spinner("Finding faces..."):
            for f in files:
                img = load_image(f.getvalue())
                faces = match(engine.detect(img), gal, threshold)
                results.append((f.name, img, faces))
        st.session_state.result_key = key
        st.session_state.results = results
    results = st.session_state.results

    # --- combine photos: best evidence per student, and faces that need a decision
    auto: dict[int, float] = {}
    review, unknown = [], []           # (photo index, face index, face)
    for p, (_, img, faces) in enumerate(results):
        for i, f in enumerate(faces):
            if f.student_id is not None:
                auto[f.student_id] = max(auto.get(f.student_id, 0), f.similarity)
            elif f.candidates and f.candidates[0][1] >= REVIEW_FLOOR:
                review.append((p, i, f))
            else:
                unknown.append((p, i, f))
    n_faces = sum(len(r[2]) for r in results)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Faces found", n_faces)
    m2.metric("Recognised", len(auto))
    m3.metric("Need a check", len(review))
    m4.metric("Unknown faces", len(unknown))

    tabs = st.tabs([f"Photo {k + 1}" for k in range(len(results))])
    for tab, (name, img, faces) in zip(tabs, results):
        with tab:
            st.image(cv2.cvtColor(annotate(img, faces, names), cv2.COLOR_BGR2RGB), caption=name)

    # --- teacher decisions for unsure / unknown faces
    confirmed: dict[int, float] = {}
    manual_faces: set[int] = set()
    unassigned = [sid for sid in names if sid not in auto]
    if review or unknown:
        st.subheader("Check these faces")
        st.caption("Orange boxes in the photo. These faces weren't a confident match, so nobody is "
                   "marked present from them unless you pick a name. Where the model has a guess, "
                   "it's shown under the face and listed first.")
        cols = st.columns(6)
        for k, (p, i, f) in enumerate(review + unknown):
            img = results[p][1]
            is_review = k < len(review)
            # Default is always "not a student": a stranger who resembles an absent
            # student must never mark them present without the teacher saying so.
            # The model's best guess is listed first so accepting it is one click.
            suggestion = [f.candidates[0][0]] if is_review and f.candidates[0][0] in unassigned else []
            options = [NOT_A_STUDENT] + suggestion + [sid for sid in unassigned if sid not in suggestion]
            default = 0
            with cols[k % 6]:
                st.image(face_thumbnail(img, f, 160), caption=f"Photo {p + 1}, face {i + 1}"
                         + (f" · maybe {names[f.candidates[0][0]]} ({f.candidates[0][1]:.2f})"
                            + (", already matched to another face" if f.candidates[0][0] in auto else "")
                            if is_review else ""))
                choice = st.selectbox("Who is this?", options, index=default,
                                      key=f"pick_{key}_{p}_{i}", label_visibility="collapsed",
                                      format_func=lambda o: o if o == NOT_A_STUDENT else names[o])
                if choice != NOT_A_STUDENT:
                    if is_review and choice == f.candidates[0][0]:
                        confirmed[choice] = f.candidates[0][1]
                    else:
                        manual_faces.add(choice)

    # --- final register: teacher can still tick / untick anyone
    st.subheader("Register")
    rows = []
    for r in roster:
        sid = r["id"]
        if sid in auto:
            status, how, sim = True, "auto", auto[sid]
        elif sid in confirmed:
            status, how, sim = True, "confirmed", confirmed[sid]
        elif sid in manual_faces:
            status, how, sim = True, "manual", None
        else:
            status, how, sim = False, "auto", None
        rows.append({"id": sid, "Present": status, "Roll": r["roll_no"], "Name": r["name"],
                     "How": {"auto": "matched" if status else "not seen",
                             "confirmed": "you confirmed", "manual": "you picked"}[how],
                     "Similarity": sim, "_how": how})
    proposal = pd.DataFrame(rows)
    edited = st.data_editor(
        proposal.drop(columns=["_how"]), hide_index=True, disabled=["Roll", "Name", "How", "Similarity"],
        column_order=["Present", "Roll", "Name", "How", "Similarity"],
        column_config={"id": None,
                       "Present": st.column_config.CheckboxColumn("Present"),
                       "Similarity": st.column_config.NumberColumn(format="%.2f")},
        key=f"editor_{key}_{hashlib.md5(proposal.to_json().encode()).hexdigest()}")

    present_n = int(edited.Present.sum())
    st.write(f"**{present_n} present, {len(edited) - present_n} absent** out of {len(edited)}.")
    if st.button("Save attendance", type="primary"):
        marks = {}
        final = dict(zip(edited["id"], edited["Present"]))        # match rows by student id
        for _, prop in proposal.iterrows():
            now_present = bool(final[prop.id])
            changed = bool(prop.Present) != now_present
            status = "present" if now_present else "absent"
            method = "manual" if changed else prop._how
            sim = None if changed or pd.isna(prop.Similarity) else float(prop.Similarity)
            marks[int(prop.id)] = (status, method, sim)
        with conn() as con:
            db.save_session(con, cls, day, subject.strip(), marks, n_faces, len(unknown))
        st.success(f"Saved {day:%d %b %Y}{' - ' + subject if subject else ''}: "
                   f"{present_n} present, {len(edited) - present_n} absent.")


def page_students():
    st.title("Students")
    with conn() as con:
        existing = db.classes(con)
    c1, c2 = st.columns([2, 3])
    with c1:
        choice = st.selectbox("Class", existing + ["+ New class"], key="stu_class") if existing \
            else "+ New class"
    with c2:
        cls = st.text_input("New class name", placeholder="e.g. BSc CSIT - Semester 3") \
            if choice == "+ New class" else choice
    if not cls:
        st.info("Type a class name to start enrolling students.")
        return

    with st.expander("Add a student", expanded=not existing or choice == "+ New class"):
        with st.form("add_student", clear_on_submit=True):
            a, b = st.columns([1, 3])
            roll = a.text_input("Roll no.")
            name = b.text_input("Full name")
            photos = st.file_uploader("1-3 clear photos of the student's face", type=["jpg", "jpeg", "png", "webp"],
                                      accept_multiple_files=True)
            cam = st.camera_input("...or take one with the webcam", key="cam")
            force = st.checkbox("Add even if this face looks like someone already enrolled")
            ok = st.form_submit_button("Add student", type="primary")
        if ok:
            add_student(cls, roll, name, [p.getvalue() for p in photos] + ([cam.getvalue()] if cam else []), force)

    with st.expander("Add a whole class at once (photo files named like 01_Sita Sharma.jpg)"):
        bulk = st.file_uploader("Select all photos", type=["jpg", "jpeg", "png", "webp"],
                                accept_multiple_files=True, key="bulk")
        if bulk and st.button("Enroll these students"):
            groups: dict[tuple[str, str], list[bytes]] = {}
            for f in bulk:
                stem = Path(f.name).stem
                roll, _, nm = stem.partition("_")
                nm = nm.rsplit("_", 1)[0] if nm.rsplit("_", 1)[-1].isdigit() else nm   # 01_Sita_2.jpg
                groups.setdefault((roll.strip(), nm.replace("_", " ").strip() or roll), []).append(f.getvalue())
            for (roll, nm), imgs in groups.items():
                add_student(cls, roll, nm, imgs, force=False, quiet=True)
            st.success(f"Processed {len(groups)} students. Check the roster below for any warnings.")

    with conn() as con:
        rows = db.roster(con, cls)
    st.subheader(f"Roster ({len(rows)})")
    if not rows:
        st.caption("Nobody enrolled in this class yet.")
        return
    cols = st.columns(6)
    for k, r in enumerate(rows):
        with cols[k % 6]:
            if r["thumb"]:
                st.image(r["thumb"], width=110)
            else:
                st.markdown(":grey[no photo]")
            st.markdown(f"**{r['roll_no']}** {r['name']}  \n:grey[{r['n_photos']} photo(s)]")
            with st.popover("Edit"):
                extra = st.file_uploader("Add another photo", type=["jpg", "jpeg", "png", "webp"],
                                         key=f"more_{r['id']}")
                if extra and st.button("Save photo", key=f"savep_{r['id']}"):
                    add_photos(r["id"], cls, [extra.getvalue()])
                    st.rerun()
                st.divider()
                if st.button("Delete student and all their data", key=f"del_{r['id']}", type="secondary"):
                    with conn() as con:
                        db.delete_student(con, r["id"])
                    st.rerun()


def _embed_all(images: list[bytes]):
    engine = get_engine()
    out, problems = [], []
    for k, data in enumerate(images, 1):
        try:
            img = load_image(data)
            emb, face = engine.enrollment_embedding(img)
            out.append((emb, face_thumbnail(img, face)))
        except ValueError as e:
            problems.append(f"Photo {k}: {e}")
    return out, problems


def add_student(cls, roll, name, images, force, quiet=False):
    if not roll or not name:
        st.error("Enter both a roll number and a name.")
        return
    if not images:
        st.error(f"{roll} {name}: add at least one photo.")
        return
    refs, problems = _embed_all(images)
    for p in problems:
        st.warning(f"{roll} {name} - {p}")
    if not refs:
        st.error(f"{roll} {name} was not added: no usable face photo.")
        return
    with conn() as con:
        gal = db.gallery(con, cls)
        names = {r["id"]: f"{r['roll_no']} {r['name']}" for r in db.roster(con, cls)}
        lookalike = [(names[sid], float((g @ e).max())) for e, _ in refs for sid, g in gal.items()
                     if float((g @ e).max()) >= threshold]
        if lookalike and not force:
            who, s = max(lookalike, key=lambda t: t[1])
            st.error(f"{roll} {name} was not added: the face matches **{who}** ({s:.2f}). "
                     "If they really are different people, tick the box and add again. "
                     "Lookalikes (e.g. twins) may need a teacher check on attendance days.")
            return
        try:
            sid = db.add_student(con, roll, name, cls)
        except sqlite3.IntegrityError:
            st.error(f"Roll no. {roll} already exists in {cls}.")
            return
        for emb, thumb in refs:
            db.add_face_ref(con, sid, emb, thumb)
    if not quiet:
        st.success(f"Added {roll} {name} with {len(refs)} photo(s).")


def add_photos(student_id, cls, images):
    refs, problems = _embed_all(images)
    for p in problems:
        st.warning(p)
    with conn() as con:
        for emb, thumb in refs:
            db.add_face_ref(con, student_id, emb, thumb)


def page_dashboard():
    st.title("Attendance dashboard")
    cls = class_picker(key="dash_class")
    if cls is None:
        st.info("No classes yet. Switch to the demo class in the sidebar to explore the dashboard.")
        return
    with conn() as con:
        ov = A.overview(con, cls)
        studs = A.student_summary(con, cls)
        weekly = A.weekly_trend(con, cls)
        wday = A.by_weekday(con, cls)
        sessions = A.session_list(con, cls)
        quality = A.recognition_quality(con, cls)
        reg = A.register(con, cls)
    if not ov["sessions"]:
        st.info("No attendance saved for this class yet.")
        return

    at_risk = studs[studs.rate < A.AT_RISK]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Average attendance", f"{ov['attendance_rate']:.0%}")
    k2.metric("Classes held", int(ov["sessions"]))
    k3.metric(f"Students below {A.AT_RISK:.0%}", f"{len(at_risk)} of {int(ov['students'])}")
    k4.metric("Marked automatically", f"{ov['auto_share']:.0%}",
              help="Share of 'present' marks the app made without the teacher changing anything.")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Attendance by week**")
        base = alt.Chart(weekly).encode(
            x=alt.X("week_start:T", title=None, axis=alt.Axis(format="%d %b", grid=False)),
            y=alt.Y("rate:Q", title=None, scale=alt.Scale(domain=[0.5, 1], zero=False), axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("week_start:T", title="Week of", format="%d %b %Y"),
                     alt.Tooltip("rate:Q", title="Attendance", format=".1%"),
                     alt.Tooltip("sessions:Q", title="Classes")])
        rule = alt.Chart(pd.DataFrame({"y": [A.AT_RISK]})).mark_rule(color=GREY, strokeDash=[4, 4]).encode(y="y:Q")
        st.altair_chart(rule + base.mark_line(color=BLUE, strokeWidth=2) + base.mark_circle(color=BLUE, size=60),
                        use_container_width=True)
        st.caption("Dashed line: the 75% attendance rule.")
    with right:
        st.markdown("**By day of the week**")
        st.altair_chart(alt.Chart(wday).mark_bar(color=BLUE, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=22)
                        .encode(x=alt.X("day:N", sort=list(wday.day), title=None, axis=alt.Axis(labelAngle=0)),
                                y=alt.Y("rate:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                                tooltip=["day", alt.Tooltip("rate:Q", format=".1%", title="Attendance"),
                                         alt.Tooltip("sessions:Q", title="Classes")]),
                        use_container_width=True)

    st.subheader("Students who need attention")
    flagged = studs[(studs.rate < A.AT_RISK) | (studs.current_absence_streak >= 3)]
    if flagged.empty:
        st.success("Everyone is above 75% and nobody has missed 3 classes in a row.")
    else:
        st.caption(f"Below {A.AT_RISK:.0%} overall, or absent from the last 3+ classes in a row.")
        st.dataframe(_student_table(flagged), hide_index=True, column_config=_student_cols())

    with st.expander(f"All students ({len(studs)})"):
        st.dataframe(_student_table(studs.sort_values("roll_no")), hide_index=True, column_config=_student_cols())

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Classes")
        show = sessions.rename(columns={"session_date": "Date", "subject": "Subject", "present": "Present",
                                        "enrolled": "Enrolled", "rate": "Attendance",
                                        "unknown_faces": "Unknown faces", "confirmed": "Confirmed",
                                        "manual": "Changed by hand"})
        show["Attendance"] = show["Attendance"] * 100
        st.dataframe(show[["Date", "Subject", "Present", "Enrolled", "Attendance", "Unknown faces",
                           "Confirmed", "Changed by hand"]], hide_index=True, height=300,
                     column_config={"Attendance": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)})
    with c2:
        st.subheader("How marks were made")
        q = quality.copy()
        q["How"] = q.method.map({"auto": "Automatic", "confirmed": "Teacher confirmed", "manual": "Changed by hand"}) \
            + " - " + q.status
        q["Share"] = q.marks / q.marks.sum() * 100
        st.dataframe(q[["How", "marks", "Share"]].rename(columns={"marks": "Marks"}), hide_index=True,
                     column_config={"Share": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})
        manual = q[q.method == "manual"].marks.sum()
        st.caption(f"The teacher changed {manual} of {q.marks.sum():,} marks "
                   f"({manual / q.marks.sum():.1%}). That's a real-world measure of how often the "
                   "model got it wrong.")

    st.subheader("Download")
    d1, d2 = st.columns(2)
    d1.download_button("Register (CSV)", reg.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"register_{cls}.csv".replace(" ", "_"), mime="text/csv")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        reg.to_excel(xl, sheet_name="Register", index=False)
        studs.to_excel(xl, sheet_name="Students", index=False)
        sessions.to_excel(xl, sheet_name="Classes", index=False)
    d2.download_button("Register + summaries (Excel)", buf.getvalue(),
                       file_name=f"attendance_{cls}.xlsx".replace(" ", "_"),
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with st.expander("Delete a saved class"):
        if not sessions.empty:
            opt = st.selectbox("Class", sessions.id, format_func=lambda i: " - ".join(
                str(v) for v in sessions.loc[sessions.id == i, ["session_date", "subject"]].iloc[0] if v))
            if st.button("Delete this attendance record"):
                with conn() as con:
                    db.delete_session(con, int(opt))
                st.rerun()


def _student_table(df):
    t = df.rename(columns={"roll_no": "Roll", "name": "Name", "sessions": "Classes", "attended": "Attended",
                           "rate": "Attendance", "current_absence_streak": "Missed in a row",
                           "last_seen": "Last present"}).copy()
    t["Attendance"] = t["Attendance"] * 100
    return t[["Roll", "Name", "Attendance", "Attended", "Classes", "Missed in a row", "Last present"]]


def _student_cols():
    return {"Attendance": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)}


def page_accuracy():
    st.title("How accurate is it?")
    path = ROOT / "reports" / "accuracy.json"
    if not path.exists():
        st.info("Run `python evaluate.py` to measure accuracy (it downloads a small public test set).")
        return
    r = json.loads(path.read_text(encoding="utf-8"))
    pt, cs = r["pair_test"], r["class_simulation"]
    d = min(cs["sweep"], key=lambda x: abs(x["threshold"] - threshold))
    p = min(pt["sweep"], key=lambda x: abs(x["threshold"] - threshold))
    st.caption(f"Tested on {r['dataset']['photos']} photos of {r['dataset']['people']} people "
               f"({r['dataset']['source']}). Numbers below use the current strictness, {threshold:.2f}.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Same/different pairs judged right", f"{p['accuracy']:.1%}", help=f"{pt['n_pairs']} labelled pairs")
    k2.metric("Present students found", f"{d['present_found']:.1%}", help="In 150 simulated class photos")
    k3.metric("Absent students marked present", f"{d['absent_marked_present']:.1%}")
    k4.metric("Strangers matched to a student", f"{d['stranger_matched']:.1%}")

    left, right = st.columns(2)
    with left:
        st.markdown("**How similar are two photos?**")
        st.caption("Pairs of photos of the same person (blue) score far higher than pairs of different "
                   "people (orange). The line is the current strictness.")
        e = pt["hist_edges"]
        h = pd.DataFrame({"sim": e[:-1] + e[:-1], "count": pt["hist_same"] + pt["hist_different"],
                          "pair": ["Same person"] * len(pt["hist_same"]) + ["Different people"] * len(pt["hist_different"])})
        bars = alt.Chart(h).mark_bar(opacity=0.85, width=9).encode(
            x=alt.X("sim:Q", title="Similarity", scale=alt.Scale(domain=[-0.2, 1])),
            y=alt.Y("count:Q", title="Pairs"),
            color=alt.Color("pair:N", scale=alt.Scale(domain=["Same person", "Different people"], range=[BLUE, ORANGE]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["pair", alt.Tooltip("sim:Q", title="From", format=".2f"), "count"])
        line = alt.Chart(pd.DataFrame({"x": [threshold]})).mark_rule(color="#333", strokeWidth=1.5).encode(x="x:Q")
        st.altair_chart(bars + line, use_container_width=True)
    with right:
        st.markdown("**The trade-off**")
        st.caption("Stricter matching means fewer wrong 'present' marks but more students missed.")
        sw = pd.DataFrame(cs["sweep"]).melt("threshold", var_name="m", value_name="rate")
        sw["m"] = sw.m.map({"present_found": "Present found", "absent_marked_present": "Absent marked present",
                            "stranger_matched": "Stranger matched"})
        lines = alt.Chart(sw).mark_line(strokeWidth=2).encode(
            x=alt.X("threshold:Q", title="Strictness"),
            y=alt.Y("rate:Q", title=None, axis=alt.Axis(format="%")),
            color=alt.Color("m:N", scale=alt.Scale(domain=["Present found", "Absent marked present", "Stranger matched"],
                                                   range=[BLUE, ORANGE, AQUA]), legend=alt.Legend(orient="top", title=None)),
            tooltip=["m", alt.Tooltip("threshold:Q", format=".2f"), alt.Tooltip("rate:Q", format=".1%")])
        line2 = alt.Chart(pd.DataFrame({"x": [threshold]})).mark_rule(color="#333", strokeWidth=1.5).encode(x="x:Q")
        st.altair_chart(lines + line2, use_container_width=True)
    st.warning("These are clean, well-lit photos of 13 people. A real classroom with 40-60 students, "
               "small faces at the back, glasses, masks or bad light will be harder. Always look over "
               "the register before saving. The 'Changed by hand' numbers on the dashboard show how the "
               "app does with your own class.")


pg = st.navigation([
    st.Page(page_take_attendance, title="Take attendance", icon=":material/photo_camera:", default=True),
    st.Page(page_students, url_path="students", title="Students", icon=":material/group:"),
    st.Page(page_dashboard, url_path="dashboard", title="Dashboard", icon=":material/insights:"),
    st.Page(page_accuracy, url_path="accuracy", title="Accuracy", icon=":material/verified:"),
])
pg.run()
