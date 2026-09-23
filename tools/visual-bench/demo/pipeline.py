"""Coordinate retrieval and candidate verification for one observation."""

import json
import time
import cv2
from .contracts import CandidateSearch, ImageMatcher
from .geometry import match_reference


def pose(value):
    return {key: value[key] for key in ("position_enu_m", "eye_to_enu_xyzw")}


def refine(
    worker, matcher, candidate, camera, query, output, sequence, candidate_id, iteration
):
    name = f"frame-{sequence:06d}-candidate-{candidate_id}-refine-{iteration}"
    reference = output / (name + "-reference.png")
    rendered = worker.request(
        dict(
            op="render",
            candidate_id=candidate_id,
            candidate=pose(candidate),
            output=str(reference),
        )
    )
    payload, metrics = match_reference(
        matcher, reference, query, reference.with_suffix(".depth.bin"), camera
    )
    matches = output / (name + "-matches.json")
    matches.write_text(json.dumps(payload))
    result = worker.request(
        dict(
            op="refine",
            candidate_id=candidate_id,
            matches=str(matches),
            output=str(output / (name + "-estimate.jsonl")),
        )
    )["estimate"]
    result.update(
        matching=metrics,
        render_ms=rendered["render_ms"],
        reference=reference.name,
        reference_pose=pose(candidate),
    )
    return result


def evaluate_candidates(worker, candidates, refine_candidate):
    """A new refinement replaces one candidate; it is not another observation."""
    attempts = []
    for candidate_id, item in enumerate(candidates):
        candidate = item["candidate"]
        for iteration in range(2):
            result = refine_candidate(candidate, candidate_id, iteration)
            attempts.append(result)
            if not result["accepted"]:
                break
            candidate = pose(result)
    result = worker.request(dict(op="select"))["observation"]
    result["geometric_attempts"] = attempts
    return result


def process_frame(
    worker,
    search: CandidateSearch,
    matcher: ImageMatcher,
    image,
    sequence,
    stamp,
    prior,
    camera,
    output,
    previous_candidates,
):
    started = time.monotonic()
    query = output / f"frame-{sequence:06d}-query.png"
    if not cv2.imwrite(str(query), image):
        raise OSError(f"Cannot save {query}")
    worker.request(
        dict(
            op="begin",
            query=str(query),
            prior=prior,
            sequence=sequence,
            capture_time_ns=stamp,
        )
    )
    candidates = [dict(candidate=pose(value)) for value in previous_candidates]
    metrics = dict(
        mode="previous_visual_proposals", stage="retrieval_only", search_seconds=0.0
    )
    evaluate = lambda proposals: evaluate_candidates(
        worker,
        proposals,
        lambda candidate, candidate_id, iteration: refine(
            worker,
            matcher,
            candidate,
            camera,
            query,
            output,
            sequence,
            candidate_id,
            iteration,
        ),
    )
    if candidates:
        result = evaluate(candidates)
    if not candidates or not any(h["accepted"] for h in result["candidate_hypotheses"]):
        candidates, metrics = search.candidates(image)
        # Keep attempt paths unique if a temporal proposal failed.
        offset = len(previous_candidates)
        result = evaluate_candidates(
            worker,
            candidates[:3],
            lambda candidate, candidate_id, iteration: refine(
                worker,
                matcher,
                candidate,
                camera,
                query,
                output,
                sequence,
                candidate_id + offset,
                iteration,
            ),
        )
    (output / f"frame-{sequence:06d}-search.json").write_text(
        json.dumps(dict(metrics=metrics, candidates=candidates), indent=2)
    )
    result.update(
        query=query.name,
        search=metrics,
        pipeline_seconds=time.monotonic() - started,
        clock_domain="input-relative-pts",
        search_scope="bounded proposals; untested locations remain possible",
    )
    return result
