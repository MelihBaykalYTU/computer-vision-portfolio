"""2026 AI-assisted synthetic planar matching companion, not VEGA source/results."""
import argparse
import csv
import hashlib
import json
import platform
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class Config:
    # Fixed before either split was evaluated; no development or test tuning.
    orb_features: int = 1800
    ratio: float = 0.75
    ransac_px: float = 3.0
    min_matches: int = 12
    min_inliers: int = 10
    min_inlier_ratio: float = 0.45
    min_reference_coverage: float = 0.18
    min_scene_area_fraction: float = 0.005
    max_scene_area_fraction: float = 0.85
    localization_tolerance_px: float = 8.0


CONFIG = Config()
REFERENCE_SEED = 20261008
DEV_CASES = [(1001, "affine"), (1002, "perspective"), (1003, "occlusion"),
             (1004, "tiny_blur"), (1101, "other_target"), (1102, "repeated"),
             (1103, "fragment"), (1104, "featureless")]
TEST_CASES = [(seed, kind) for kind, seeds in [
    ("affine", (2001, 2002, 2003)), ("perspective", (2011, 2012, 2013)),
    ("occlusion", (2021, 2022, 2023)), ("tiny_blur", (2031, 2032, 2033)),
    ("other_target", (2101, 2102, 2103, 2104)), ("repeated", (2111, 2112, 2113)),
    ("fragment", (2121, 2122, 2123, 2124)), ("featureless", (2131,))]
    for seed in seeds]
POSITIVE_KINDS = {"affine", "perspective", "occlusion", "tiny_blur"}
PROVENANCE = (
    "AI-assisted educational companion created on 2026-10-08. New synthetic "
    "experiment, not original VEGA competition source, not historical competition "
    "performance, and not evidence of the candidate's individual proficiency."
)


def corners(shape):
    h, w = shape[:2]
    return np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])


def project(points, matrix):
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    if not np.isfinite(homogeneous).all() or np.any(np.abs(homogeneous[:, 2]) < 1e-8):
        return None
    return (homogeneous[:, :2] / homogeneous[:, 2:3]).astype(np.float32)


def coverage(points, shape):
    if len(points) < 3 or not np.isfinite(points).all():
        return 0.0
    return float(cv.contourArea(cv.convexHull(np.float32(points)))) / (shape[0] * shape[1])


def geometry_guard(source, destination, reference_shape, scene_shape, matrix, mask):
    """Reject insufficient/non-planar evidence before accepting a quadrilateral."""
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        return "homography_failed", None
    inliers = mask.ravel().astype(bool)
    if np.count_nonzero(inliers) < CONFIG.min_inliers:
        return "too_few_inliers", None
    if np.mean(inliers) < CONFIG.min_inlier_ratio:
        return "low_inlier_ratio", None
    # Spatial support in both images rules out line/point fits and tiny fragments.
    if coverage(source[inliers], reference_shape) < CONFIG.min_reference_coverage:
        return "low_reference_coverage", None
    if coverage(destination[inliers], scene_shape) < CONFIG.min_scene_area_fraction / 4:
        return "degenerate_scene_support", None
    quad = project(corners(reference_shape), matrix)
    if quad is None or not cv.isContourConvex(quad.reshape(-1, 1, 2)):
        return "invalid_projected_quad", None
    # A reference plane must not cross the homography's line at infinity.
    denom = np.column_stack((corners(reference_shape), np.ones(4))) @ matrix[2]
    if not (np.all(denom > 1e-8) or np.all(denom < -1e-8)):
        return "projective_horizon", None
    area = cv.contourArea(quad, oriented=True)
    if area <= 0:
        return "reflected_or_collapsed_quad", None
    area_fraction = area / (scene_shape[0] * scene_shape[1])
    if not CONFIG.min_scene_area_fraction <= area_fraction <= CONFIG.max_scene_area_fraction:
        return "implausible_quad_area", None
    h, w = scene_shape[:2]
    if np.any(quad < [-0.25 * w, -0.25 * h]) or np.any(quad > [1.25 * w, 1.25 * h]):
        return "quad_outside_scene", None
    return "accepted", quad


class Matcher:
    def __init__(self, reference):
        if not isinstance(reference, np.ndarray) or reference.dtype != np.uint8 or reference.ndim != 2 or reference.size == 0:
            raise ValueError("Reference must be a non-empty uint8 grayscale image.")
        self.shape = reference.shape
        self.orb = cv.ORB_create(nfeatures=CONFIG.orb_features, fastThreshold=12)
        self.keypoints, self.descriptors = self.orb.detectAndCompute(reference, None)
        if self.descriptors is None or len(self.keypoints) < CONFIG.min_matches:
            raise ValueError("Reference has too few usable ORB features.")
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

    def match(self, scene, seed):
        start = time.perf_counter()
        result = {"accepted": False, "reason": "invalid_scene", "keypoints": 0,
                  "matches": 0, "inliers": 0, "inlier_ratio": 0.0,
                  "reference_coverage": 0.0, "quad": None}
        if not isinstance(scene, np.ndarray) or scene.dtype != np.uint8 or scene.ndim != 2 or scene.size == 0:
            result["match_ms"] = (time.perf_counter() - start) * 1000
            return result
        kp, descriptors = self.orb.detectAndCompute(scene, None)
        result["keypoints"] = len(kp)
        if descriptors is None or len(kp) < 2:
            result["reason"] = "too_few_scene_features"
        else:
            pairs = self.bf.knnMatch(self.descriptors, descriptors, k=2)
            good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < CONFIG.ratio * pair[1].distance]
            # Each scene keypoint contributes at most once, avoiding inflated support.
            good = sorted(good, key=lambda m: (m.distance, m.queryIdx, m.trainIdx))
            used, unique = set(), []
            for match in good:
                if match.trainIdx not in used:
                    unique.append(match)
                    used.add(match.trainIdx)
            result["matches"] = len(unique)
            if len(unique) < CONFIG.min_matches:
                result["reason"] = "too_few_ratio_matches"
            else:
                src = np.float32([self.keypoints[m.queryIdx].pt for m in unique])
                dst = np.float32([kp[m.trainIdx].pt for m in unique])
                if coverage(src, self.shape) < CONFIG.min_reference_coverage:
                    result["reason"] = "low_candidate_coverage"
                else:
                    cv.setRNGSeed(seed)
                    matrix, mask = cv.findHomography(src, dst, cv.RANSAC, CONFIG.ransac_px,
                                                     maxIters=2000, confidence=0.995)
                    if mask is not None:
                        inside = mask.ravel().astype(bool)
                        result.update(inliers=int(np.count_nonzero(inside)),
                                      inlier_ratio=float(np.mean(inside)),
                                      reference_coverage=coverage(src[inside], self.shape))
                    reason, quad = geometry_guard(src, dst, self.shape, scene.shape, matrix, mask)
                    result.update(accepted=reason == "accepted", reason=reason,
                                  quad=None if quad is None else quad.tolist())
        result["match_ms"] = (time.perf_counter() - start) * 1000
        return result


def texture(seed, size=(240, 300)):
    """An original deterministic texture, with independent shapes and glyphs."""
    rng = np.random.default_rng(seed)
    h, w = size
    image = np.full((h, w), 218, np.uint8)
    for i in range(110):
        x, y = rng.integers(10, w - 10), rng.integers(10, h - 10)
        value = int(rng.integers(10, 195))
        if i % 3:
            cv.circle(image, (int(x), int(y)), int(rng.integers(2, 10)), value, int(rng.choice([-1, 1, 2])))
        else:
            end = (int(np.clip(x + rng.integers(-25, 26), 0, w - 1)),
                   int(np.clip(y + rng.integers(-25, 26), 0, h - 1)))
            cv.line(image, (int(x), int(y)), end, value, 2)
    alphabet = "ACGKLNRSXYZ2346789"
    for row in range(4):
        for col in range(5):
            text = alphabet[int(rng.integers(len(alphabet)))]
            cv.putText(image, text, (12 + col * 58, 46 + row * 52), cv.FONT_HERSHEY_SIMPLEX,
                       0.75, int(rng.integers(0, 130)), 2, cv.LINE_AA)
    return image


def background(rng):
    image = np.clip(68 + np.linspace(0, 25, 640)[None, :] + rng.normal(0, 2, (480, 640)), 0, 255).astype(np.uint8)
    for _ in range(18):
        a = tuple(int(x) for x in rng.integers([0, 0], [640, 480]))
        b = tuple(int(x) for x in rng.integers([0, 0], [640, 480]))
        cv.line(image, a, b, int(rng.integers(45, 130)), 1)
    return image


def paste(scene, target, quad):
    matrix = cv.getPerspectiveTransform(corners(target.shape), np.float32(quad))
    warped = cv.warpPerspective(target, matrix, (scene.shape[1], scene.shape[0]))
    mask = cv.warpPerspective(np.full(target.shape, 255, np.uint8), matrix, (scene.shape[1], scene.shape[0]), flags=cv.INTER_NEAREST)
    scene[mask > 0] = warped[mask > 0]
    return matrix


def make_case(reference, seed, kind):
    rng = np.random.default_rng(seed)
    scene = background(rng)
    matrix = None
    if kind in POSITIVE_KINDS:
        scale = rng.uniform(0.75, 1.05) if kind != "tiny_blur" else rng.uniform(0.24, 0.36)
        angle = rng.uniform(-30, 30)
        transform = cv.getRotationMatrix2D((149.5, 119.5), angle, scale)
        transform[:, 2] += [rng.uniform(130, 190), rng.uniform(75, 125)]
        quad = project(corners(reference.shape), np.vstack((transform, [0, 0, 1])))
        if kind == "perspective":
            quad += rng.uniform(-40, 40, (4, 2)).astype(np.float32)
        matrix = paste(scene, reference, quad)
        if kind == "occlusion":
            # Occlude 30-50% of the target width with a vertical foreground strip.
            lo, hi = quad.min(axis=0), quad.max(axis=0)
            strip_width = (hi[0] - lo[0]) * rng.uniform(0.30, 0.50)
            x = int(lo[0] + rng.uniform(0.12, 0.35) * (hi[0] - lo[0]))
            cv.rectangle(scene, (x, int(lo[1]) - 3), (int(x + strip_width), int(hi[1]) + 3), 78, -1)
            cv.line(scene, (x, int(lo[1])), (int(x + strip_width), int(hi[1])), 105, 3)
        if kind == "affine":
            scene = np.clip(scene.astype(np.float32) * rng.uniform(0.75, 1.15) + rng.uniform(-12, 12), 0, 255).astype(np.uint8)
        if kind == "tiny_blur":
            scene = cv.GaussianBlur(scene, (7, 7), 1.8)
    elif kind == "other_target":
        for i, shift in enumerate((0, 220)):
            other = texture(seed + 500 + i)
            paste(scene, other, np.float32([[35 + shift, 75], [245 + shift, 92],
                                          [235 + shift, 267], [44 + shift, 256]]))
    elif kind == "repeated":
        tile = np.full((55, 55), 210, np.uint8)
        cv.putText(tile, "X8", (1, 35), cv.FONT_HERSHEY_SIMPLEX, 0.8, 30, 2, cv.LINE_AA)
        pattern = np.tile(tile, (5, 7))
        scene[90:365, 120:505] = pattern
    elif kind == "fragment":
        # A fragment is deliberately labelled absent: the complete reference is missing.
        x, y = int(rng.integers(20, 165)), int(rng.integers(20, 125))
        fragment = reference[y:y + 80, x:x + 95]
        paste(scene, fragment, np.float32([[210, 170], [343, 155], [351, 267], [219, 282]]))
    elif kind == "featureless":
        scene = np.full((480, 640), 90, np.uint8)
    else:
        raise ValueError(kind)
    truth = None if matrix is None else project(corners(reference.shape), matrix)
    return scene, truth


def sanity_checks(reference, matcher):
    """Independent known-coordinate and rejected-input checks, not score mirroring."""
    translation = np.array([[1., 0., 13.], [0., 1., -7.], [0., 0., 1.]])
    sample = np.float32([[2, 4], [8, 10], [20, 30]])
    np.testing.assert_allclose(project(sample, translation), sample + [13, -7], atol=1e-5)
    blank = np.full((480, 640), 100, np.uint8)
    assert matcher.match(blank, 77)["reason"] == "too_few_scene_features"
    assert matcher.match(np.empty((0, 0), np.uint8), 77)["reason"] == "invalid_scene"
    for unusable in (np.empty((0, 0), np.uint8), np.full((100, 100), 80, np.uint8)):
        try:
            Matcher(unusable)
        except ValueError:
            pass
        else:
            raise AssertionError("An unusable reference must fail clearly.")
    line = np.float32([[i * 10., 50.] for i in range(14)])
    reason, _ = geometry_guard(line, line + 30, reference.shape, blank.shape, np.eye(3), np.ones((14, 1), np.uint8))
    assert reason == "low_reference_coverage"
    grid = np.float32([[x, y] for x in (20, 90, 180, 275) for y in (20, 100, 210)])
    reason, _ = geometry_guard(grid, grid, reference.shape, blank.shape, np.full((3, 3), np.nan), np.ones((12, 1), np.uint8))
    assert reason == "homography_failed"
    # A separate fixture with independent known corners verifies direction and point order.
    scene = np.full((480, 640), 70, np.uint8)
    truth = np.float32([[135, 90], [450, 115], [427, 360], [120, 340]])
    paste(scene, reference, truth)
    found = matcher.match(scene, 8888)
    assert found["accepted"], found
    error = float(np.mean(np.linalg.norm(np.array(found["quad"]) - truth, axis=1)))
    assert error < CONFIG.localization_tolerance_px, error
    unrelated, _ = make_case(reference, 98765, "other_target")
    assert not matcher.match(unrelated, 9999)["accepted"]
    return {"checks_passed": 9, "known_fixture_corner_error_px": error,
            "checks": ["translation coordinates", "featureless scene", "empty scene",
                       "empty reference", "featureless reference", "collinear correspondences",
                       "nonfinite homography", "independent known quadrilateral", "unrelated targets"]}


def summarize(rows):
    tp = sum(r["positive"] and r["accepted"] for r in rows)
    fp = sum(not r["positive"] and r["accepted"] for r in rows)
    fn = sum(r["positive"] and not r["accepted"] for r in rows)
    tn = sum(not r["positive"] and not r["accepted"] for r in rows)
    errors = [r["corner_error_px"] for r in rows if r["corner_error_px"] is not None]
    durations = [r["match_ms"] for r in rows]
    return {"cases": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": None if tp + fp == 0 else tp / (tp + fp),
            "recall": None if tp + fn == 0 else tp / (tp + fn),
            "localized_positive_count": sum(e <= CONFIG.localization_tolerance_px for e in errors),
            "accepted_positive_corner_error_px_mean": float(np.mean(errors)) if errors else None,
            "accepted_positive_corner_error_px_max": max(errors) if errors else None,
            "match_ms_median": float(np.median(durations)), "match_ms_p95": float(np.percentile(durations, 95)),
            "rejection_reasons": dict(Counter(r["reason"] for r in rows if not r["accepted"]))}


def overview(examples, reference, path):
    tiles = []
    for row, scene, truth in examples:
        tile = cv.cvtColor(scene, cv.COLOR_GRAY2BGR)
        if truth is not None:
            cv.polylines(tile, [np.int32(np.round(truth))], True, (40, 205, 40), 3)
        if row["quad"] is not None:
            cv.polylines(tile, [np.int32(np.round(row["quad"]))], True, (30, 130, 255), 2)
        tile = cv.resize(tile, (480, 360), interpolation=cv.INTER_AREA)
        footer = np.full((63, 480, 3), 250, np.uint8)
        status = "PRESENT" if row["positive"] else "ABSENT"
        decision = "accepted" if row["accepted"] else "rejected"
        cv.putText(footer, f'{row["kind"]} #{row["seed"]}: {status}, {decision}', (8, 21), cv.FONT_HERSHEY_SIMPLEX, 0.48, (35, 35, 35), 1, cv.LINE_AA)
        error = "" if row["corner_error_px"] is None else f' | corner error {row["corner_error_px"]:.2f}px'
        cv.putText(footer, row["reason"] + error, (8, 46), cv.FONT_HERSHEY_SIMPLEX, 0.44, (35, 35, 35), 1, cv.LINE_AA)
        tiles.append(np.vstack((tile, footer)))
    header = np.full((88, 960, 3), 250, np.uint8)
    thumb = cv.cvtColor(cv.resize(reference, (90, 72)), cv.COLOR_GRAY2BGR)
    header[8:80, 8:98] = thumb
    cv.putText(header, "Synthetic planar reference matching: held-out examples", (112, 29), cv.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 1, cv.LINE_AA)
    cv.putText(header, "Green: known target corners | Orange: accepted estimate", (112, 55), cv.FONT_HERSHEY_SIMPLEX, 0.49, (35, 35, 35), 1, cv.LINE_AA)
    cv.putText(header, "New AI-assisted 2026 demo; not VEGA source or competition results", (112, 76), cv.FONT_HERSHEY_SIMPLEX, 0.43, (35, 35, 35), 1, cv.LINE_AA)
    canvas = np.vstack((header, *[np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))
    if not cv.imwrite(str(path), canvas):
        raise OSError("Could not write overview image.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cv.setNumThreads(1)
    cv.ocl.setUseOpenCL(False)
    assert not ({s for s, _ in DEV_CASES} & {s for s, _ in TEST_CASES})
    assert len({s for s, _ in DEV_CASES + TEST_CASES}) == len(DEV_CASES + TEST_CASES)
    started = time.perf_counter()
    reference = texture(REFERENCE_SEED)
    prepare_started = time.perf_counter()
    matcher = Matcher(reference)
    prepare_ms = (time.perf_counter() - prepare_started) * 1000
    checks = sanity_checks(reference, matcher)
    rows, examples = [], []
    # Test fixtures never enter the parameter selection or sanity-check code.
    for split, cases in (("development", DEV_CASES), ("test", TEST_CASES)):
        for seed, kind in cases:
            scene, truth = make_case(reference, seed, kind)
            found = matcher.match(scene, seed)
            error = None
            if truth is not None and found["accepted"]:
                error = float(np.mean(np.linalg.norm(np.array(found["quad"]) - truth, axis=1)))
            row = {"split": split, "seed": seed, "kind": kind, "positive": truth is not None,
                   **found, "corner_error_px": error}
            rows.append(row)
            if split == "test" and kind not in {e[0]["kind"] for e in examples}:
                examples.append((row, scene, truth))
    summary = {"provenance": PROVENANCE, "reference_seed": REFERENCE_SEED,
               "config": asdict(CONFIG), "parameter_selection": "Fixed a priori; no tuning on development or test outcomes.",
               "reference_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
               "runtime": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv.__version__,
                           "platform": platform.platform(), "cpu_threads": 1, "opencl": False},
               "sanity": checks, "reference_prepare_ms": prepare_ms,
               "splits": {split: summarize([r for r in rows if r["split"] == split]) for split in ("development", "test")},
               "test_by_kind": {kind: summarize([r for r in rows if r["split"] == "test" and r["kind"] == kind])
                                for kind in sorted({k for _, k in TEST_CASES})},
               "cases": rows}
    overview(examples, reference, args.output / "overview.png")
    summary["total_run_seconds"] = time.perf_counter() - started
    (args.output / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fields = [key for key in rows[0] if key != "quad"]
    with (args.output / "cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for split, stats in summary["splits"].items():
        print(f"{split}: TP={stats['tp']} FP={stats['fp']} FN={stats['fn']} TN={stats['tn']} "
              f"precision={stats['precision']:.3f} recall={stats['recall']:.3f} "
              f"median={stats['match_ms_median']:.1f}ms")
    print(f"Sanity checks: {checks['checks_passed']} passed; outputs: {args.output}")


if __name__ == "__main__":
    main()
