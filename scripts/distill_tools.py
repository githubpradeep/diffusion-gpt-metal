"""Distill a tool-call dataset for the looped model.

A strong teacher writes the user request. The call itself is fixed before
that request is written, so the label cannot drift. Every argument value
has to appear in the request. Rows that fail that check are dropped.

The teacher is whatever OPENROUTER_API_KEY can call. Output is JSONL plus
a plain text rendering the trainer can read later.

  data/tools/calls.jsonl
  data/tools/calls.txt
"""

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "openrouter/free"

TOOLS = [
    "set_timer",
    "set_alarm",
    "add_reminder",
    "send_message",
    "create_event",
    "get_weather",
    "play_music",
    "navigate",
    "add_note",
    "open_app",
]

CONTACTS = [
    "Priya", "Arjun", "Meera", "Rahul", "Anita", "Vikram", "Sara", "Dev",
    "Leela", "Omar", "Nina", "Kabir", "Rosa", "Ethan", "Aisha", "Jon",
    "Maya", "Luis", "Hana", "Chris",
]
CITIES = [
    "Pune", "Delhi", "Mumbai", "Bengaluru", "Jaipur", "Kochi", "London",
    "Paris", "Tokyo", "Seattle", "Austin", "Cairo", "Lisbon", "Oslo",
]
APPS = [
    "Maps", "Calendar", "Notes", "Music", "Messages", "Camera", "Clock",
    "Weather", "Mail", "Photos", "Podcasts", "Settings",
]
SONGS = [
    "Yellow", "Hummingbird", "River", "Midnight", "Postcard", "Lantern",
    "Golden", "Espresso", "Satellite", "Paper Plane",
]
ARTISTS = ["Coldplay", "Adele", "Arijit", "Sufjan", "Beatles", "Dua Lipa", "Kishore"]
LABELS = ["pasta", "tea", "laundry", "eggs", "homework", "stretch", "bread", "call"]
NOTES = [
    "buy milk", "gate code 4412", "idea for the talk", "pick up the parcel",
    "dentist on Friday", "password hint river", "return the book",
]
REMINDERS = [
    "take the medicine", "water the plants", "submit the form",
    "charge the laptop", "feed the cat", "send the invoice",
]
EVENTS = [
    "dentist", "design review", "lunch with Meera", "football",
    "piano lesson", "team standup",
]
PLACES = [
    "the station", "home", "the clinic", "Pune airport", "the library",
    "Dadar", "the office", "Koregaon Park",
]
WHENS = ["today", "tomorrow", "tonight", "Friday", "Monday morning", "this evening"]
DATES = ["tomorrow", "Friday", "Monday", "March 3", "April 12", "next Sunday"]
TIMES = ["7:05", "7:30", "8:15", "9:00", "12:40", "6:10", "6:45", "8:20", "10:30"]
STYLES = [
    "a short fragment",
    "a full polite sentence",
    "a hurried text",
    "a spoken request",
    "a slightly messy sentence",
]


def minutes():
    return random.choice([1, 2, 3, 5, 7, 8, 10, 12, 15, 17, 20, 25, 30, 45, 90])


def spec_for(tool):
    if tool == "set_timer":
        args = {"minutes": str(minutes())}
        if random.random() < 0.6:
            args["label"] = random.choice(LABELS)
        return args
    if tool == "set_alarm":
        args = {"time": random.choice(TIMES)}
        if random.random() < 0.5:
            args["label"] = random.choice(LABELS)
        return args
    if tool == "add_reminder":
        return {"text": random.choice(REMINDERS), "when": random.choice(WHENS)}
    if tool == "send_message":
        return {"contact": random.choice(CONTACTS), "text": random.choice(NOTES)}
    if tool == "create_event":
        return {
            "title": random.choice(EVENTS),
            "date": random.choice(DATES),
            "time": random.choice(TIMES),
        }
    if tool == "get_weather":
        args = {"location": random.choice(CITIES)}
        if random.random() < 0.55:
            args["when"] = random.choice(WHENS)
        return args
    if tool == "play_music":
        args = {"title": random.choice(SONGS)}
        if random.random() < 0.55:
            args["artist"] = random.choice(ARTISTS)
        return args
    if tool == "navigate":
        return {"destination": random.choice(PLACES)}
    if tool == "add_note":
        return {"text": random.choice(NOTES)}
    if tool == "open_app":
        return {"app": random.choice(APPS)}
    raise KeyError(tool)


def plan(count, seed, refusals):
    rng_state = random.getstate()
    random.seed(seed)
    rows = []
    per = count // len(TOOLS)
    extra = count % len(TOOLS)
    for i, tool in enumerate(TOOLS):
        n = per + (1 if i < extra else 0)
        for _ in range(n):
            args = spec_for(tool)
            rows.append({
                "kind": "call",
                "tool": tool,
                "arguments": args,
                "style": random.choice(STYLES),
            })
    for _ in range(refusals):
        rows.append({
            "kind": "refuse",
            "tool": None,
            "arguments": {},
            "style": random.choice([
                "a joke request",
                "a math question",
                "a news question",
                "a recipe request",
                "an opinion question",
                "a translation request",
            ]),
        })
    random.shuffle(rows)
    random.setstate(rng_state)
    return rows


def grounded(text, arguments):
    hay = text.casefold()
    for value in arguments.values():
        if str(value).casefold() not in hay:
            return False
    return True


def render(query, tool, arguments):
    if tool is None:
        call = "<refuse>"
    else:
        parts = [tool]
        for key in arguments:
            parts.append(key)
            parts.append(str(arguments[key]))
        call = "<call> " + " ".join(parts)
    return f"<request> {query}\n{call}"


def chat(key, model, text, timeout):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "temperature": 0.9,
        "max_tokens": 1200,
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def message_text(data):
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    if isinstance(content, list):
        bits = []
        for item in content:
            if isinstance(item, str):
                bits.append(item)
            elif isinstance(item, dict):
                bits.append(item.get("text") or "")
        content = "\n".join(bits)
    return re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()


def prompt_for(batch):
    lines = [
        "Write one natural user request per item.",
        "The request must contain every value exactly as written, including digits and punctuation.",
        "Do not name the tool. Do not add values that are not in the item.",
        "One line each, starting with the item number and a period.",
        "",
    ]
    for i, row in enumerate(batch, 1):
        if row["kind"] == "refuse":
            lines.append(
                f"{i}. Write {row['style']} that is NOT a timer, alarm, reminder, "
                "message, calendar event, weather, song, navigation, note, or app launch."
            )
        else:
            facts = ", ".join(f"{k}={v}" for k, v in row["arguments"].items())
            lines.append(f"{i}. Style: {row['style']}. Facts that must appear: {facts}")
    return "\n".join(lines)


def parse_lines(text, n):
    found = {}
    for line in text.splitlines():
        match = re.match(r"^\s*(\d+)\s*[\).\-:]\s*(.+)$", line.strip())
        if not match:
            continue
        idx = int(match.group(1))
        query = match.group(2).strip().strip('"')
        if 1 <= idx <= n and query:
            found[idx] = query
    return found


def load_done(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_text(path, rows):
    lines = [render(row["query"], row["tool"], row["arguments"]) for row in rows]
    path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=4000)
    parser.add_argument("--refusals", type=int, default=500)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--out", default="data/tools/calls.jsonl")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    seen = {row["query"].strip() for row in done}
    jobs = plan(args.count, args.seed, args.refusals)
    # Resume by count, not by matching specs. Already-written rows stay.
    jobs = jobs[len(done):]
    print(f"have {len(done)}  left {len(jobs)}  model {args.model}", flush=True)

    empty = 0
    while jobs:
        batch = jobs[: args.batch]
        del jobs[: args.batch]
        text = ""
        routed = ""
        for attempt in range(6):
            try:
                data = chat(key, args.model, prompt_for(batch), args.timeout)
                if "error" in data:
                    raise RuntimeError(json.dumps(data["error"])[:240])
                routed = str(data.get("model") or "")
                text = message_text(data)
                if parse_lines(text, len(batch)):
                    break
                text = ""
                time.sleep(2)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode()[:200]
                wait = 15 * (attempt + 1) if exc.code == 429 else 6
                print(f"http {exc.code} wait {wait}s {detail}", flush=True)
                time.sleep(wait)
            except Exception as exc:
                print(f"retry {attempt} {exc}", flush=True)
                time.sleep(5)
        parsed = parse_lines(text, len(batch))
        kept = 0
        missed = []
        with out.open("a", encoding="utf-8") as handle:
            for i, row in enumerate(batch, 1):
                query = parsed.get(i, "")
                ok = bool(query) and query not in seen
                if ok and row["kind"] == "call" and not grounded(query, row["arguments"]):
                    ok = False
                if not ok:
                    missed.append(row)
                    continue
                seen.add(query)
                item = {
                    "query": query,
                    "tool": row["tool"],
                    "arguments": row["arguments"],
                }
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                done.append(item)
                kept += 1
        if missed and kept:
            jobs.extend(missed)
        if kept == 0:
            empty += 1
            jobs = batch + jobs
            print(f"kept 0 via {routed} empty {empty}", flush=True)
            if text:
                print(text[:300], flush=True)
            if empty >= 6:
                print("stopping after empty replies", flush=True)
                break
            time.sleep(8)
        else:
            empty = 0
            print(f"rows {len(done)}  kept {kept}/{len(batch)}  via {routed}", flush=True)
            if len(done) % 80 < kept:
                write_text(out.with_suffix(".txt"), done)

    write_text(out.with_suffix(".txt"), done)
    calls = sum(1 for row in done if row["tool"])
    refusals = len(done) - calls
    print(f"wrote {out}  calls {calls}  refusals {refusals}")


if __name__ == "__main__":
    main()
