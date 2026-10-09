"""
How accurate is the face matching? Two tests, written to reports/accuracy.json.

Test data: the public test images from the open-source deepface library
(github.com/serengil/deepface, MIT licence): 61 photos of 13 public figures with
528 labelled "same person / different person" pairs. They are downloaded into
eval_data/ (not part of this project) the first time you run this.

1. Pair test - for every labelled pair, compute the similarity and check whether the
   threshold separates "same" from "different". Gives ROC AUC, accuracy at each
   threshold, and the best threshold.
2. Classroom simulation - pretend some of the 13 people are enrolled students (one
   enrollment photo each) and the rest are strangers. Build synthetic class photos
   (a collage of other photos of whoever "attended", plus strangers), run the full
   pipeline, and count: students correctly marked present, absent students wrongly
   marked present, and strangers wrongly matched to a student.

Run:  python evaluate.py
"""
from __future__ import annotations

import json
import random
import urllib.request
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from attendance.engine import DEFAULT_THRESHOLD, FaceEngine, Face, load_image, match

ROOT = Path(__file__).parent
EVAL_DIR = ROOT / "eval_data"
REPORT = ROOT / "reports" / "accuracy.json"
BASE = "https://raw.githubusercontent.com/serengil/deepface/master/tests/unit/dataset/"
SEED = 7
THRESHOLDS = np.round(np.arange(0.20, 0.61, 0.01), 2)


# ------------------------------------------------------------------ data
def download() -> pd.DataFrame:
    EVAL_DIR.mkdir(exist_ok=True)
    for name in ["master.csv", "face-recognition-pivot.csv"]:
        if not (EVAL_DIR / name).exists():
            urllib.request.urlretrieve(BASE + name, EVAL_DIR / name)
    m = pd.read_csv(EVAL_DIR / "master.csv").rename(
        columns={"file_x": "x", "file_y": "y", "Decision": "decision"})
    p = pd.read_csv(EVAL_DIR / "face-recognition-pivot.csv")
    p["x"], p["y"] = p.file_x.str.split("/").str[-1], p.file_y.str.split("/").str[-1]
    pairs = pd.concat([m[["x", "y", "decision"]], p[["x", "y", "decision"]]]).drop_duplicates()
    pairs["same"] = pairs.decision.eq("Yes")
    for img in sorted(set(pairs.x) | set(pairs.y)):
        if not (EVAL_DIR / img).exists():
            print("downloading", img)
            urllib.request.urlretrieve(BASE + img, EVAL_DIR / img)
    return pairs.reset_index(drop=True)


def identities(pairs: pd.DataFrame) -> list[list[str]]:
    """Group photos into people using the 'same person' pairs (union-find)."""
    parent: dict[str, str] = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in pairs[pairs.same][["x", "y"]].values:
        parent[find(a)] = find(b)
    for a in set(pairs.x) | set(pairs.y):
        find(a)
    groups: dict[str, list[str]] = {}
    for a in parent:
        groups.setdefault(find(a), []).append(a)
    # sanity check: no "different person" pair may fall inside one group
    label = {img: k for k, g in groups.items() for img in g}
    bad = pairs[~pairs.same & (pairs.x.map(label) == pairs.y.map(label))]
    assert bad.empty, f"inconsistent labels: {bad}"
    return sorted((sorted(g) for g in groups.values()), key=len, reverse=True)


# ------------------------------------------------------------------ test 1
def pair_test(engine: FaceEngine, pairs: pd.DataFrame) -> tuple[dict, dict]:
    faces: dict[str, Face] = {}
    for img in sorted(set(pairs.x) | set(pairs.y)):
        found = engine.detect(load_image(EVAL_DIR / img))
        faces[img] = max(found, key=lambda f: f.box[2] * f.box[3])  # main subject
    sims = np.array([faces[a].embedding @ faces[b].embedding for a, b in pairs[["x", "y"]].values])
    same = pairs.same.values

    # ROC AUC = chance a random "same" pair scores above a random "different" pair
    pos, neg = sims[same], sims[~same]
    auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    sweep = []
    for t in THRESHOLDS:
        pred = sims >= t
        sweep.append({"threshold": float(t),
                      "accuracy": float((pred == same).mean()),
                      "true_match_rate": float(pred[same].mean()),     # same person accepted
                      "false_match_rate": float(pred[~same].mean())})  # different person accepted
    best = max(sweep, key=lambda r: (r["accuracy"], -abs(r["threshold"] - DEFAULT_THRESHOLD)))
    at_default = min(sweep, key=lambda r: abs(r["threshold"] - DEFAULT_THRESHOLD))
    hist_edges = np.round(np.arange(-0.2, 1.01, 0.05), 2)
    result = {
        "n_pairs": int(len(pairs)), "n_same": int(same.sum()), "n_different": int((~same).sum()),
        "auc": auc, "best": best, "at_default": at_default, "sweep": sweep,
        "hist_edges": hist_edges.tolist(),
        "hist_same": np.histogram(pos, hist_edges)[0].tolist(),
        "hist_different": np.histogram(neg, hist_edges)[0].tolist(),
        "min_same": float(pos.min()), "max_different": float(neg.max()),
    }
    return result, faces


# ------------------------------------------------------------------ test 2
def crop_face(img: np.ndarray, face: Face, size: int) -> np.ndarray:
    """Head-and-shoulders crop around a face, resized so the face is ~size px tall."""
    x, y, w, h = face.box
    cx, cy = x + w / 2, y + h / 2
    half = int(max(w, h) * 1.1)
    H, W = img.shape[:2]
    x0, y0 = max(0, int(cx - half)), max(0, int(cy - half))
    x1, y1 = min(W, int(cx + half)), min(H, int(cy + half))
    crop = img[y0:y1, x0:x1]
    scale = size / max(1, h)
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def collage(tiles: list[np.ndarray], cols: int, rng: random.Random) -> np.ndarray:
    """Arrange people in rows like a class photo (random order, grey background)."""
    rng.shuffle(tiles)
    cell = max(max(t.shape[:2]) for t in tiles) + 10
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.full((rows * cell, cols * cell, 3), 200, np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        y = r * cell + (cell - t.shape[0]) // 2
        x = c * cell + (cell - t.shape[1]) // 2
        canvas[y:y + t.shape[0], x:x + t.shape[1]] = t
    return canvas


def class_simulation(engine: FaceEngine, people: list[list[str]], faces: dict[str, Face],
                     n_classes: int = 150) -> dict:
    rng = random.Random(SEED)
    eligible = [p for p in people if len(p) >= 2]   # need 1 enrollment + 1 class photo
    images = {img: load_image(EVAL_DIR / img) for p in people for img in p}
    counts = {t: dict(tp=0, fn=0, fp_absent=0, wrong_student=0, stranger_fp=0, strangers=0)
              for t in THRESHOLDS}
    n_present = n_absent = 0
    examples = []
    for k in range(n_classes):
        rng.shuffle(eligible)
        enrolled = eligible[:9]                                  # 9 "students"
        strangers = [p for p in people if p not in enrolled]     # everyone else
        gallery, photos = {}, {}
        for sid, person in enumerate(enrolled):
            enroll_img, *rest = rng.sample(person, len(person))
            gallery[sid] = faces[enroll_img].embedding[None, :]
            photos[sid] = rest
        present = [sid for sid in gallery if rng.random() < 0.8]  # ~80% turn up
        visitors = rng.sample(strangers, min(len(strangers), rng.randint(0, 2)))
        tiles, truth = [], []
        for sid in present:
            img = rng.choice(photos[sid])
            tiles.append(crop_face(images[img], faces[img], rng.randint(45, 140)))
            truth.append(sid)
        for person in visitors:
            img = rng.choice(person)
            tiles.append(crop_face(images[img], faces[img], rng.randint(45, 140)))
        if not tiles:
            continue
        photo = collage(tiles, cols=rng.randint(3, 5), rng=random.Random(k))
        detected = engine.detect(photo)
        n_present += len(present)
        n_absent += len(gallery) - len(present)
        for t in THRESHOLDS:
            marked = {f.student_id for f in match(detected, gallery, float(t)) if f.student_id is not None}
            c = counts[t]
            c["tp"] += len(set(present) & marked)
            c["fn"] += len(set(present) - marked)
            c["fp_absent"] += len(marked - set(present))
        if k < 3:
            examples.append({"faces_in_photo": len(tiles), "faces_detected": len(detected)})
        # strangers: how often is a visitor's face matched to some student?
        for person in visitors:
            img = rng.choice(person)
            emb = faces[img].embedding
            best = max(float((gallery[s] @ emb).max()) for s in gallery)
            for t in THRESHOLDS:
                counts[t]["strangers"] += 1
                counts[t]["stranger_fp"] += int(best >= t)

    sweep = []
    for t, c in counts.items():
        sweep.append({"threshold": float(t),
                      "present_found": c["tp"] / max(1, n_present),
                      "absent_marked_present": c["fp_absent"] / max(1, n_absent),
                      "stranger_matched": c["stranger_fp"] / max(1, c["strangers"])})
    at = lambda thr: min(sweep, key=lambda r: abs(r["threshold"] - thr))
    return {"n_classes": n_classes, "students_per_class": 9,
            "present_student_slots": n_present, "absent_student_slots": n_absent,
            "sweep": sweep, "at_default": at(DEFAULT_THRESHOLD), "examples": examples}


def main() -> None:
    pairs = download()
    people = identities(pairs)
    print(f"{sum(map(len, people))} photos of {len(people)} people, {len(pairs)} labelled pairs")
    engine = FaceEngine()

    pt, faces = pair_test(engine, pairs)
    print(f"\nPair test: AUC {pt['auc']:.3f}")
    print(f"  at threshold {pt['at_default']['threshold']}: accuracy {pt['at_default']['accuracy']:.1%}, "
          f"same-person accepted {pt['at_default']['true_match_rate']:.1%}, "
          f"different-person accepted {pt['at_default']['false_match_rate']:.1%}")
    print(f"  best threshold {pt['best']['threshold']}: accuracy {pt['best']['accuracy']:.1%}")

    sim = class_simulation(engine, people, faces)
    d = sim["at_default"]
    print(f"\nClassroom simulation ({sim['n_classes']} synthetic class photos, threshold {d['threshold']}):")
    print(f"  present students found:        {d['present_found']:.1%}")
    print(f"  absent students marked present: {d['absent_marked_present']:.1%}")
    print(f"  strangers matched to a student: {d['stranger_matched']:.1%}")

    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps({
        "model": "YuNet (detection) + SFace (recognition), OpenCV Model Zoo",
        "default_threshold": DEFAULT_THRESHOLD,
        "dataset": {"source": "deepface unit-test images (MIT licence)",
                    "photos": sum(map(len, people)), "people": len(people),
                    "pairs": len(pairs)},
        "pair_test": pt, "class_simulation": sim}, indent=1), encoding="utf-8")
    print("\nWrote", REPORT)


if __name__ == "__main__":
    main()
