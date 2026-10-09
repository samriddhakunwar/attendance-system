"""
Face engine: finds faces in a photo and decides who they are.

Models (both from the OpenCV Model Zoo, Apache-2.0 licence):
  * YuNet  - fast face detector that also finds 5 landmarks (eyes, nose, mouth corners)
  * SFace  - turns an aligned face into a 128-number "embedding"; two photos of the
             same person give embeddings that point in nearly the same direction

Matching uses cosine similarity between embeddings (1.0 = identical direction).
Each enrolled student can have several reference embeddings; a face's score for a
student is the best of them.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

try:   # hide a harmless OpenCV notice about compute targets
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_URLS = {
    "face_detection_yunet_2023mar.onnx":
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx":
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

# Similarity needed to call a face a match. OpenCV publishes 0.363 for SFace; on our
# test data (evaluate.py) 0.40 kept every true match while cutting wrong matches, and a
# wrong "present" is worse than a miss the teacher can fix, so we use the stricter value.
DEFAULT_THRESHOLD = 0.40
# Faces scoring between REVIEW_FLOOR and the threshold are shown as "please confirm"
# with the best guess pre-selected, instead of silently becoming "unknown".
REVIEW_FLOOR = 0.30
MAX_SIDE = 2560          # very large phone photos are shrunk to this for speed
MIN_UPSCALE_SIDE = 1280  # small photos are enlarged so distant faces are still found
MIN_FACE_PX = 20         # smaller "faces" (posters, photos on walls) are too small to recognise


MODEL_SIZES = {"face_detection_yunet_2023mar.onnx": 232_589,
               "face_recognition_sface_2021dec.onnx": 38_696_353}


def _download(url: str, dest: Path, size: int, attempts: int = 4) -> None:
    """Download with retries and a size check, so a dropped connection
    never leaves a half-written model behind."""
    tmp = dest.with_suffix(".part")
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as out:
                while chunk := r.read(1 << 20):
                    out.write(chunk)
            if tmp.stat().st_size == size:
                tmp.replace(dest)
                return
            last_error = f"got {tmp.stat().st_size:,} of {size:,} bytes"
        except OSError as e:          # network errors, timeouts
            last_error = str(e)
        print(f"  attempt {attempt} failed ({last_error}), retrying...")
    tmp.unlink(missing_ok=True)
    raise RuntimeError(
        f"Could not download {dest.name} ({last_error}).\n"
        f"Download it in your browser from:\n  {url}\n"
        f"and save it as:\n  {dest}")


def ensure_models() -> dict[str, Path]:
    """Download the two ONNX models on first run (about 38 MB in total)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in MODEL_URLS.items():
        path = MODELS_DIR / name
        if not path.exists() or path.stat().st_size != MODEL_SIZES[name]:
            print(f"Downloading {name} ...")
            _download(url, path, MODEL_SIZES[name])
        paths[name] = path
    return paths


@dataclass
class Face:
    box: tuple[int, int, int, int]        # x, y, w, h in the ORIGINAL photo's pixels
    score: float                          # detector confidence 0-1
    embedding: np.ndarray                 # 128-d, unit length
    student_id: int | None = None         # filled in by match()
    similarity: float = 0.0               # best similarity to the matched student
    candidates: list[tuple[int, float]] = field(default_factory=list)  # top guesses


def load_image(data: bytes | str | Path) -> np.ndarray:
    """Read an image from bytes (an upload) or a path, as BGR uint8."""
    if isinstance(data, (str, Path)):
        data = Path(data).read_bytes()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read this file as an image (try JPG or PNG).")
    return img


class FaceEngine:
    def __init__(self, det_score: float = 0.7):
        paths = ensure_models()
        self.detector = cv2.FaceDetectorYN.create(
            str(paths["face_detection_yunet_2023mar.onnx"]), "", (320, 320),
            score_threshold=det_score, nms_threshold=0.3, top_k=5000)
        self.recognizer = cv2.FaceRecognizerSF.create(
            str(paths["face_recognition_sface_2021dec.onnx"]), "")

    # ---------------------------------------------------------------- detection
    def _detect_at(self, img: np.ndarray, scale: float) -> list[tuple[np.ndarray, float, np.ndarray]]:
        """Run the detector on a resized copy; return (detection, scale, resized image)."""
        work = img if scale == 1.0 else cv2.resize(
            img, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        self.detector.setInputSize((work.shape[1], work.shape[0]))
        _, dets = self.detector.detect(work)
        return [] if dets is None else [(d, scale, work) for d in dets]

    def detect(self, img: np.ndarray) -> list[Face]:
        """Find all faces and compute an embedding for each.

        Two passes: a large one for small, distant faces at the back of a class,
        and a small (640 px) one for big close-up faces, which the detector misses
        at full resolution. Overlapping boxes from the two passes are merged."""
        h, w = img.shape[:2]
        long_side = max(h, w)
        main = MAX_SIDE / long_side if long_side > MAX_SIDE else \
            MIN_UPSCALE_SIDE / long_side if long_side < MIN_UPSCALE_SIDE else 1.0
        found = self._detect_at(img, main)
        if long_side * main > 900:
            found += self._detect_at(img, 640 / long_side)

        # merge duplicates (same face found in both passes): keep the higher score
        found.sort(key=lambda t: -t[0][14])
        kept: list[tuple[np.ndarray, float, np.ndarray]] = []
        for d, s, work in found:
            box = d[:4] / s
            if min(box[2], box[3]) < MIN_FACE_PX:
                continue
            if all(_iou(box, k[0][:4] / k[1]) < 0.4 for k in kept):
                kept.append((d, s, work))

        faces: list[Face] = []
        for d, s, work in kept:
            aligned = self.recognizer.alignCrop(work, d)          # uses the 5 landmarks
            emb = self.recognizer.feature(aligned).flatten().astype(np.float32)
            emb /= np.linalg.norm(emb) + 1e-9
            x, y, bw, bh = (d[:4] / s).round().astype(int)
            faces.append(Face(box=(int(x), int(y), int(bw), int(bh)),
                              score=float(d[14]), embedding=emb))
        return _reading_order(faces)

    def enrollment_embedding(self, img: np.ndarray) -> tuple[np.ndarray, Face]:
        """Embedding for an enrollment photo. Requires exactly one clear face
        (if several are found, the largest is used, as in a selfie with people behind)."""
        faces = self.detect(img)
        if not faces:
            raise ValueError("No face found. Use a clear, front-facing photo with good light.")
        face = max(faces, key=lambda f: f.box[2] * f.box[3])
        return face.embedding, face


def _reading_order(faces: list[Face]) -> list[Face]:
    """Number faces like reading a page: row by row, left to right. Faces whose
    centres are within ~60% of a face height vertically count as the same row."""
    if not faces:
        return faces
    tol = 0.6 * float(np.median([f.box[3] for f in faces]))
    by_y = sorted(faces, key=lambda f: f.box[1] + f.box[3] / 2)
    rows, current = [], [by_y[0]]
    for f in by_y[1:]:
        cy = f.box[1] + f.box[3] / 2
        row_cy = np.mean([g.box[1] + g.box[3] / 2 for g in current])
        if cy - row_cy <= tol:
            current.append(f)
        else:
            rows.append(current)
            current = [f]
    rows.append(current)
    return [f for row in rows for f in sorted(row, key=lambda f: f.box[0])]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


# -------------------------------------------------------------------- matching
def similarity_matrix(faces: list[Face], gallery: dict[int, np.ndarray]) -> tuple[np.ndarray, list[int]]:
    """Rows = faces, columns = students; each cell = best cosine similarity over
    that student's reference embeddings. gallery maps student_id -> (k, 128) array."""
    ids = list(gallery)
    if not faces or not ids:
        return np.zeros((len(faces), len(ids))), ids
    F = np.stack([f.embedding for f in faces])                     # (n, 128)
    S = np.zeros((len(faces), len(ids)), dtype=np.float32)
    for j, sid in enumerate(ids):
        S[:, j] = (F @ gallery[sid].T).max(axis=1)
    return S, ids


def match(faces: list[Face], gallery: dict[int, np.ndarray],
          threshold: float = DEFAULT_THRESHOLD) -> list[Face]:
    """Assign each face to at most one student and each student to at most one face.

    Greedy one-to-one assignment: take the highest-similarity (face, student) pair,
    lock both, repeat. This stops two faces in the same photo both being marked as
    the same student (e.g. twins, or a blurry face that looks like someone else)."""
    S, ids = similarity_matrix(faces, gallery)
    for i, f in enumerate(faces):
        order = np.argsort(-S[i])[:3] if len(ids) else []
        f.candidates = [(ids[j], float(S[i, j])) for j in order]
        f.student_id, f.similarity = None, float(S[i].max()) if len(ids) else 0.0
    if not len(ids):
        return faces
    pairs = sorted(((S[i, j], i, j) for i in range(len(faces)) for j in range(len(ids))
                    if S[i, j] >= threshold), reverse=True)
    used_f, used_s = set(), set()
    for s, i, j in pairs:
        if i in used_f or j in used_s:
            continue
        faces[i].student_id, faces[i].similarity = ids[j], float(s)
        used_f.add(i)
        used_s.add(j)
    return faces


# ------------------------------------------------------------------- drawing
def face_thumbnail(img: np.ndarray, face: Face, size: int = 112) -> bytes:
    """Square JPEG crop around a face (with a little margin), for previews."""
    x, y, w, h = face.box
    cx, cy, half = x + w / 2, y + h / 2, max(w, h) * 0.7
    H, W = img.shape[:2]
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(W, cx + half)), int(min(H, cy + half))
    crop = cv2.resize(img[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()



def annotate(img: np.ndarray, faces: list[Face], names: dict[int, str]) -> np.ndarray:
    """Draw a numbered box on each face: green = matched student, orange = unknown."""
    out = img.copy()
    h, w = out.shape[:2]
    thick = max(2, round(max(h, w) / 700))
    font = max(0.5, max(h, w) / 1800)
    for n, f in enumerate(faces, 1):
        x, y, bw, bh = f.box
        known = f.student_id is not None
        color = (60, 170, 40) if known else (30, 140, 245)   # BGR
        cv2.rectangle(out, (x, y), (x + bw, y + bh), color, thick)
        label = f"{n}. {names.get(f.student_id, '?')}" if known else f"{n}. unknown"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, thick)
        ty = y - 6 if y - th - 10 > 0 else y + bh + th + 6
        tx = max(0, min(x, w - tw - 6))                 # keep the label inside the photo
        cv2.rectangle(out, (tx, ty - th - base), (tx + tw + 6, ty + base), color, -1)
        cv2.putText(out, label, (tx + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, font,
                    (255, 255, 255), max(1, thick - 1), cv2.LINE_AA)
    return out
