"""Find VRChat chatbox bubbles (and who wrote them) among OCR text lines.

VRChat always shows a player's chatbox directly ABOVE their nameplate, and the app knows who is in the
instance (VRChat log). So: (1) lines that match a player name are nameplates (fuzzy, survives OCR slips and
any nameplate style, incl. the 2026 redesign); (2) the text block directly above and centred on a nameplate is
that player's chat. Without a player list we fall back to pixels: white text whose background is clearly
darker than the scene around it (VRChat's box is translucent black), with the short line below as the name.
Pure Python/numpy so it runs inside the isolated OCR worker.
Input: lines [(text, score, box4x2)], BGR image, optional players [str].
Output: [{"name","name_guess","text","box","score"}].
"""
from __future__ import annotations

import difflib
import re

import numpy as np

_LUM = np.array([0.114, 0.587, 0.299], np.float32)


def _key(s: str) -> str:
    return re.sub(r"[\W_]+", "", s.lower())


def _match_player(text: str, players: list[str]) -> str | None:
    k = _key(text)
    if not k or len(k) > 40:
        return None
    best, best_r = None, 0.0
    for p in players:
        pk = _key(p)
        if not pk:
            continue
        r = 1.0 if pk == k else difflib.SequenceMatcher(None, k, pk).ratio()
        if r > best_r:
            best, best_r = p, r
    return best if best_r >= (0.6 if len(k) >= 4 else 0.5) else None


def _items(lines):
    out = []
    for text, score, box in lines:
        b = np.asarray(box, np.float32)
        x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 0].max(), b[:, 1].max()
        out.append({"text": text, "score": float(score), "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "cx": (x0 + x1) / 2, "h": max(1.0, y1 - y0)})
    return out


def _join_rows(items):
    """OCR sometimes cuts one visual line into side-by-side boxes: join pieces on the same baseline."""
    items = sorted(items, key=lambda it: (round(it["y0"] / 4), it["x0"]))
    out: list[dict] = []
    for it in items:
        for o in out:
            ov = min(o["y1"], it["y1"]) - max(o["y0"], it["y0"])
            gap = it["x0"] - o["x1"]
            if ov > 0.6 * min(o["h"], it["h"]) and -0.3 * o["h"] < gap < 1.6 * max(o["h"], it["h"]):
                cjk = any("　" <= c <= "鿿" or "가" <= c <= "힯" for c in o["text"] + it["text"])
                o["text"] = o["text"].rstrip() + ("" if cjk else " ") + it["text"].lstrip()
                o["x1"] = max(o["x1"], it["x1"]); o["y0"] = min(o["y0"], it["y0"]); o["y1"] = max(o["y1"], it["y1"])
                o["cx"] = (o["x0"] + o["x1"]) / 2; o["h"] = max(1.0, o["y1"] - o["y0"])
                o["score"] = min(o["score"], it["score"])
                break
        else:
            out.append(dict(it))
    return out


def _group(items):
    """Stack the lines of one text block: aligned centres, small vertical gap, similar height."""
    items = sorted(items, key=lambda it: (it["y0"], it["cx"]))
    groups: list[list[dict]] = []
    for it in items:
        for g in groups:
            last = g[-1]
            if abs(it["cx"] - last["cx"]) < max(30, 0.3 * max(last["x1"] - last["x0"], it["x1"] - it["x0"])) and \
                    -0.35 * last["h"] <= it["y0"] - last["y1"] < 0.7 * last["h"] and 0.6 < it["h"] / last["h"] < 1.6:
                g.append(it)
                break
        else:
            groups.append([it])
    out = []
    for g in groups:
        out.append({"lines": g, "x0": min(i["x0"] for i in g), "y0": min(i["y0"] for i in g),
                    "x1": max(i["x1"] for i in g), "y1": max(i["y1"] for i in g),
                    "h": float(np.median([i["h"] for i in g])), "score": float(np.mean([i["score"] for i in g]))})
    for b in out:
        b["cx"] = (b["x0"] + b["x1"]) / 2
    return out


def _block_text(block) -> str:
    cjk = any("　" <= c <= "鿿" or "가" <= c <= "힯" for i in block["lines"] for c in i["text"])
    return ("" if cjk else " ").join(i["text"].strip() for i in block["lines"])


def _dark_box(img, b) -> bool:
    """White text on a background clearly darker than the scene around the block."""
    H, W = img.shape[:2]
    h = b["h"]

    def ring(a, c, below=True):
        X0, Y0 = max(0, int(b["x0"] - c)), max(0, int(b["y0"] - c))
        X1, Y1 = min(W, int(b["x1"] + c)), min(H, int(b["y1"] + c))
        reg = img[Y0:Y1, X0:X1].astype(np.float32)
        yy, xx = np.mgrid[Y0:Y1, X0:X1]
        inner = (xx >= b["x0"] - a) & (xx <= b["x1"] + a) & (yy >= b["y0"] - a) & (yy <= b["y1"] + a)
        keep = ~inner if below else (~inner & (yy <= b["y1"] + a))
        lum = reg[keep] @ _LUM
        lum = lum[lum < 170]
        return float(np.median(lum)) if lum.size > 20 else None
    inside, outside = ring(0.06 * h, 0.28 * h), ring(0.95 * h, 1.45 * h, below=False)
    reg = img[max(0, int(b["y0"])):int(b["y1"]), max(0, int(b["x0"])):int(b["x1"])]
    white = float((reg.min(axis=2) > 175).mean()) if reg.size else 0.0
    if inside is None or white < 0.02:
        return False
    return inside < 90 and (outside is None or inside < 0.8 * outside or inside < 40)


# VRChat 2026 chatbox: opaque dark-slate rounded bubble (measured on real screenshots: BGR ~ 36,35,27, a slight
# teal tint B > R). Nameplate pills are greyer/bluer (R >= B) and dark scenery is warmer, so a tight colour match
# plus the tint separates them; a loose match leaks into dark rooms and merges the bubble with the scene.
BUBBLE_BGR = np.array([36, 35, 27], np.int16)


def find_bubble_regions(img) -> list[dict]:
    """Chat bubbles found by their own shape and colour - works with nameplates OFF. Returns contours."""
    import cv2
    im = img.astype(np.int16)
    m = (np.abs(im - BUBBLE_BGR).sum(axis=2) < 20) & ((im[..., 0] - im[..., 2]) >= 4)
    mask = m.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))      # cut thin bridges to scenery
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
    white = (img.min(axis=2) > 185)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue
        (cx, cy), (w, h), ang = cv2.minAreaRect(c)
        long, short = max(w, h), min(w, h)
        if short < 18 or short > 400 or long / max(short, 1) > 30 or area / max(w * h, 1) < 0.7:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        inside = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(inside, [c], -1, 1, -1)
        sub = inside[y:y + bh, x:x + bw].astype(bool)
        wt = white[y:y + bh, x:x + bw][sub].mean() if sub.any() else 0.0
        if wt < 0.01 or wt > 0.45:                  # needs white text, but isn't a white panel
            continue
        out.append({"contour": c, "x0": x, "y0": y, "x1": x + bw, "y1": y + bh, "h": short})
    return out


# OSC "status" lines that chatbox apps keep re-sending (clock, now playing, song progress, heart rate, ...).
# They update in place every few seconds, so they are shown but never translated or read aloud.
_STATUS = [re.compile(p, re.I) for p in (
    r"^[\W_]*\d{1,2}\s*[:.：]\s*\d{2}(\s*[:.]\s*\d{2})?\s*([ap]\.?\s?m\.?)?[\W_]*$",    # 12:00 PM
    r"\d{1,2}[:.]\d{2}\s*[/|]\s*\d{1,2}[:.]\d{2}",                                        # 2:23 / 3:42
    r"^[\W_\d\s/|:.]{1,8}$",                                                               # '3142', '| 3:42'
    r"^\s*[♪♫♬🎵🎶🎧]",                                                                       # now playing
    r"\b\d{2,3}\s*bpm\b|[❤♥💓]\s*\d{2,3}\b|\bhr\s*[:=]?\s*\d{2,3}\b",                      # heart rate
    r"\b(now playing|listening to|spotify|youtube music)\b",
    r"^\s*(cpu|gpu|ram|fps)\b.*\d",
    r"^.{1,60}\sby\s.{1,40}\s\d{1,2}[:.]?\d{2}\s*$",                                        # Song by Artist 2:23
)]
_SONG = re.compile(r"^.{1,60}\s(by|-|–)\s.{1,40}$")


def split_status(lines: list[str]) -> tuple[list[str], list[str]]:
    """Split a bubble's lines into (message lines, status lines)."""
    status = [any(r.search(t) for r in _STATUS) for t in lines]
    if any(status):          # a status bubble: a bare 'Song by Artist' line belongs to it too
        status = [s or bool(_SONG.match(t)) for s, t in zip(status, lines)]
    msg = [re.sub(r"^[\s💭🗨💬]+", "", t) for t, s in zip(lines, status) if not s]
    return [t for t in msg if t], [t for t, s in zip(lines, status) if s]


def _inside(reg, it) -> bool:
    import cv2
    return cv2.pointPolygonTest(reg["contour"], (float(it["cx"]), float((it["y0"] + it["y1"]) / 2)), False) >= 0


def _nameplate_below(box, cands, players, used):
    best, best_d = None, 1e9
    for k, t in enumerate(cands):
        if k in used:
            continue
        gap = t["y0"] - box["y1"]
        if not (-0.2 * box["h"] < gap < 5.0 * max(box["h"], t["h"])):
            continue
        if abs(t["cx"] - (box["x0"] + box["x1"]) / 2) > max(90, 0.7 * (box["x1"] - box["x0"])):
            continue
        name = _match_player(t["text"], players) if players else None
        if players and not name:
            continue
        d = gap + 0.5 * abs(t["cx"] - (box["x0"] + box["x1"]) / 2)
        if d < best_d:
            best, best_d = (k, name), d
    return best


def find_chatboxes(lines, img, players: list[str] | None = None) -> list[dict]:
    """Chat bubbles + who wrote them. Order of trust:
    1. bubble shapes (2026 style; works with nameplates off), named by a matching nameplate below if visible;
    2. nameplate-anchored blocks (player list) for bubbles the shape pass missed;
    3. classic dark-box fallback when there is neither a player list nor any bubble shape."""
    items = _join_rows(_items(lines))
    players = [p for p in (players or []) if p]
    out, taken = [], set()
    regions = find_bubble_regions(img)
    for reg in regions:
        mine = [k for k, it in enumerate(items) if k not in taken and _inside(reg, it)]
        if not mine:
            continue
        taken.update(mine)
        bl = sorted((items[k] for k in mine), key=lambda it: (it["y0"], it["x0"]))
        block = {"lines": bl, "x0": reg["x0"], "y0": reg["y0"], "x1": reg["x1"], "y1": reg["y1"],
                 "h": float(np.median([i["h"] for i in bl]))}
        rest = [it for k, it in enumerate(items) if k not in taken]
        nb = _nameplate_below(block, rest, players, set())
        name = ""
        if nb is not None:
            name = nb[1] or ""
        msg, status = split_status([i["text"].strip() for i in bl])
        if not msg and not status or not re.search(r"\w\w", " ".join(msg + status)):
            continue                                 # an empty/typing bubble ('•')
        block["lines"] = [i for i in bl if i["text"].strip() in msg]
        out.append({"name": name, "name_guess": not bool(nb and nb[1]), "text": _block_text(block) if msg else "",
                    "status": status, "lines": msg,
                    "box": [float(reg["x0"]), float(reg["y0"]), float(reg["x1"]), float(reg["y1"])],
                    "score": float(np.mean([i["score"] for i in bl]))})
    left = [it for k, it in enumerate(items) if k not in taken]
    items = left
    if players:
        tags, rest = [], []
        for it in items:
            m = _match_player(it["text"], players)
            (tags if m else rest).append((it, m) if m else it)
        blocks = _group(rest)
        used = set()
        for tag, name in sorted(tags, key=lambda t: t[0]["y0"]):
            best, best_d = None, 1e9
            for k, b in enumerate(blocks):
                gap = tag["y0"] - b["y1"]                     # bubble bottom -> nameplate top
                if k in used or not (-0.2 * b["h"] < gap < 5.0 * max(b["h"], tag["h"])):
                    continue
                if abs(b["cx"] - tag["cx"]) > max(80, 0.6 * (b["x1"] - b["x0"])):
                    continue
                d = gap + 0.5 * abs(b["cx"] - tag["cx"])
                if d < best_d:
                    best, best_d = k, d
            if best is None:
                continue
            used.add(best)
            b = blocks[best]
            out.append({"name": name, "name_guess": False, "text": _block_text(b),
                        "lines": [i["text"].strip() for i in b["lines"]],
                        "box": [float(b["x0"]), float(b["y0"]), float(b["x1"]), float(b["y1"])], "score": b["score"]})
        # With a player list we only trust bubbles anchored on a nameplate: VRChat's own menus (Launch Pad,
        # HUD hints, emoji panel) are white-on-dark too and would otherwise read as chat.
        return out
    if out:
        return out          # 2026-style bubbles found: don't guess from other dark UI text
    # no player list and no bubble shapes: classic dark-box fallback
    blocks = _group(items)
    used = set()
    for k, b in enumerate(blocks):
        if not _dark_box(img, b):
            continue
        best, best_d = None, 1e9
        for j, t in enumerate(blocks):
            gap = t["y0"] - b["y1"]
            if j == k or j in used or not (0 <= gap < 4.5 * b["h"]) or abs(t["cx"] - b["cx"]) > max(60, (b["x1"] - b["x0"]) / 2):
                continue
            if len(_block_text(t)) > 32:
                continue
            d = gap + 0.5 * abs(t["cx"] - b["cx"])
            if d < best_d:
                best, best_d = j, d
        name = ""
        if best is not None:
            used.add(best)
            name = _block_text(blocks[best])
        out.append({"name": name, "name_guess": True, "text": _block_text(b),
                    "lines": [i["text"].strip() for i in b["lines"]], "box": [float(b["x0"]), float(b["y0"]), float(b["x1"]), float(b["y1"])], "score": b["score"]})
    names = {o["name"] for o in out if o["name"]}
    return [o for o in out if o["text"] not in names]
