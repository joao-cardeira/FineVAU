import json

DIMS = ("what", "who", "where")
PAPER_WEIGHTS = {"what": 1.0, "who": 2.0, "where": 1.0}
FIELDS = [f"{d}_{k}" for d in DIMS for k in ("obt", "tot")]


def _int_in(x, allowed):
    if isinstance(x, bool):
        x = int(x)
    if isinstance(x, (int, float)) and float(x) == int(x) and int(x) in allowed:
        return int(x)
    return None


def attribute_score(v):
    if isinstance(v, list):
        items = [_int_in(x, (0, 1)) for x in v]
        if not items or None in items:
            return None
        return 1 if 2 * sum(items) >= len(items) else 0
    return _int_in(v, (0, 1))


def valid_scores(raw):
    if not isinstance(raw, dict) or not all(k in raw for k in ("location", "entity", "event")):
        return False
    if not isinstance(raw["location"], dict) or not isinstance(raw["entity"], list) or not isinstance(raw["event"], list):
        return False
    if any(_int_in(v, (0, 1)) is None for v in raw["location"].values()):
        return False
    for e in raw["entity"]:
        if not isinstance(e, dict) or _int_in(e.get("presence_score"), (0, 1)) is None:
            return False
        attrs = e.get("attributes", {})
        if not isinstance(attrs, dict) or any(attribute_score(v) is None for v in attrs.values()):
            return False
    return all(isinstance(e, dict) and _int_in(e.get("score"), (0, 1, 2)) is not None for e in raw["event"])


def parse_reply(text):
    t = (text or "").strip()
    if t.startswith("```"):
        parts = t.split("```")
        t = parts[1] if len(parts) > 1 else ""
        if t.startswith("json"):
            t = t[4:]
        t = t.strip()
    try:
        raw = json.loads(t)
    except ValueError:
        return None
    return raw if valid_scores(raw) else None


def strip_reasoning(output, thinking_opened):
    if "</think>" in output:
        return output.rsplit("</think>", 1)[1].strip()
    if thinking_opened or "<think>" in output:
        return None
    return output.strip()


def recall_counts(raw):
    out = dict.fromkeys(FIELDS, 0)
    for v in raw["location"].values():
        out["where_obt"] += int(v)
        out["where_tot"] += 1
    for e in raw["entity"]:
        out["who_obt"] += int(e["presence_score"])
        out["who_tot"] += 1
        for v in e.get("attributes", {}).values():
            out["who_obt"] += attribute_score(v)
            out["who_tot"] += 1
    for e in raw["event"]:
        out["what_obt"] += int(e["score"]) / 2
        out["what_tot"] += 1
    return out


def zero_counts(ground_truth):
    entities = ground_truth.get("Entity") or []
    return {
        "what_obt": 0, "what_tot": len(ground_truth.get("Event") or []),
        "who_obt": 0, "who_tot": sum(1 + len([k for k in e if k not in ("id", "category")]) for e in entities),
        "where_obt": 0, "where_tot": len(ground_truth.get("Location") or {}),
    }


def aggregate(rows, weights=PAPER_WEIGHTS):
    rows = list(rows)
    by_video = {}
    for n, r in enumerate(rows):
        by_video.setdefault(r.get("video_id", n), []).append(r)

    means = []
    for answers in by_video.values():
        scored = [a["counts"] for a in answers if a.get("counts")]
        if scored:
            means.append({f: sum(c[f] for c in scored) / len(scored) for f in FIELDS})

    dims = {}
    for d in DIMS:
        obtained = sum(m[f"{d}_obt"] for m in means)
        total = sum(m[f"{d}_tot"] for m in means)
        dims[d] = {"obtained": obtained, "total": total, "recall": 100 * obtained / total if total else None}
    num = sum(weights[d] * dims[d]["obtained"] for d in DIMS)
    den = sum(weights[d] * dims[d]["total"] for d in DIMS)
    scored = sum(1 for r in rows if r.get("counts"))
    no_answer = sum(1 for r in rows if r.get("status") == "no_answer")
    return {
        "videos": len(by_video),
        "videos_failed": len(by_video) - len(means),
        "answers": len(rows),
        "answers_per_video": max((len(a) for a in by_video.values()), default=0),
        "judged": scored - no_answer,
        "no_answer": no_answer,
        "judge_failed": len(rows) - scored,
        "weights": dict(weights),
        "dimensions": dims,
        "fv_score": 100 * num / den if den else None,
    }


def format_report(agg):
    line = "-" * 62
    out = [
        "", line, f"  FV-Score (mean@{agg['answers_per_video']})", line,
        f"  videos {agg['videos']}   answers {agg['answers']}",
        f"  judged {agg['judged']}   no answer {agg['no_answer']}   judge failed {agg['judge_failed']}",
        line, f"  {'dimension':<10} {'weight':>7} {'obtained':>10} {'total':>10} {'recall':>9}", line,
    ]
    for d in DIMS:
        v = agg["dimensions"][d]
        recall = "n/a" if v["recall"] is None else f"{v['recall']:.1f}%"
        out.append(f"  {d:<10} {agg['weights'][d]:>7g} {v['obtained']:>10.2f} {v['total']:>10.2f} {recall:>9}")
    fv = agg["fv_score"]
    out += [line, f"  FV-Score   {'n/a' if fv is None else f'{fv:.2f}'}", line]
    if agg["judge_failed"]:
        out.append(f"  {agg['judge_failed']} answers left out: the judge never returned a usable reply")
    if agg["videos_failed"]:
        out.append(f"  {agg['videos_failed']} videos left out: none of their answers could be scored")
    return "\n".join(out + [""])
