#!/usr/bin/env python3

import argparse
import csv
import json
import os
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda iterable: iterable


VENUE_ID = "ICLR.cc/2026/Conference"

DEFAULT_ACCEPTED_VENUE_IDS = [
    f"{VENUE_ID}/Accepted_Submission",
    f"{VENUE_ID}/Oral",
    f"{VENUE_ID}/Spotlight",
    f"{VENUE_ID}/Poster",
]
DEFAULT_REJECTED_VENUE_ID = f"{VENUE_ID}/Rejected_Submission"
DEFAULT_DESK_REJECTED_VENUE_ID = f"{VENUE_ID}/Desk_Rejected_Submission"


def get_value(content, key, default=None):
    if not content:
        return default

    value = content.get(key, default)
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def make_client(username=None, password=None):
    try:
        import openreview
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: openreview-py. Install it with "
            "`pip install openreview-py tqdm`."
        ) from exc

    kwargs = {"baseurl": "https://api2.openreview.net"}
    if username and password:
        kwargs.update({"username": username, "password": password})
    return openreview.api.OpenReviewClient(**kwargs)


def unique(values):
    out = []
    seen = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def discover_venue_ids(client):
    group = client.get_group(VENUE_ID)
    content = group.content or {}

    accepted = []
    rejected = []
    desk_rejected = []

    for key, raw_value in content.items():
        value = raw_value.get("value") if isinstance(raw_value, dict) else raw_value
        if not isinstance(value, str):
            continue

        lower_key = key.lower()
        lower_value = value.lower()

        if "desk" in lower_key or "desk_rejected" in lower_value:
            desk_rejected.append(value)
        elif "reject" in lower_key or "rejected_submission" in lower_value:
            rejected.append(value)
        elif "accept" in lower_key or "accepted_submission" in lower_value:
            accepted.append(value)

    return {
        "accepted": unique(accepted + DEFAULT_ACCEPTED_VENUE_IDS),
        "rejected": unique(rejected + [DEFAULT_REJECTED_VENUE_ID]),
        "desk_rejected": unique(desk_rejected + [DEFAULT_DESK_REJECTED_VENUE_ID]),
    }


def fetch_notes_for_venue_ids(client, venue_ids):
    notes_by_id = {}
    used_venue_ids = []

    for venue_id in venue_ids:
        try:
            notes = client.get_all_notes(content={"venueid": venue_id})
        except Exception as exc:
            print(f"[FAIL] {venue_id}: {exc}")
            continue

        if notes:
            print(f"[OK] {venue_id}: {len(notes)} papers")
            used_venue_ids.append(venue_id)
            for note in notes:
                notes_by_id[note.id] = note
        else:
            print(f"[EMPTY] {venue_id}")

    return list(notes_by_id.values()), used_venue_ids


def is_desk_reject(note, desk_rejected_venue_ids):
    venueid = str(get_value(note.content, "venueid", "") or "")
    venue = str(get_value(note.content, "venue", "") or "")
    combined = f"{venueid} {venue}".lower()

    return venueid in desk_rejected_venue_ids or "desk rejected" in combined


def paper_to_record(note):
    return {
        "paper_id": note.id,
        "number": getattr(note, "number", None),
        "title": get_value(note.content, "title", ""),
        "abstract": get_value(note.content, "abstract", ""),
        "primary_area": get_value(note.content, "primary_area", ""),
        "url": f"https://openreview.net/forum?id={note.id}",
    }


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path, records):
    fieldnames = ["paper_id", "number", "title", "abstract", "primary_area", "url"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download ICLR 2026 accepted or rejected paper metadata from "
            "OpenReview, excluding desk rejects."
        )
    )
    parser.add_argument(
        "--decision",
        choices=["accept", "reject"],
        required=True,
        help="Which class to download. One run can download only one class.",
    )
    parser.add_argument(
        "--out-dir",
        default="iclr2026_papers",
        help="Output directory. Default: iclr2026_papers.",
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "csv"],
        default="jsonl",
        help="Output file format. Default: jsonl.",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Download only the first N papers. Useful for testing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    username = os.getenv("OPENREVIEW_USERNAME")
    password = os.getenv("OPENREVIEW_PASSWORD")
    client = make_client(username, password)

    print("[1] Discovering ICLR 2026 venue ids...")
    venue_ids = discover_venue_ids(client)
    desk_rejected_venue_ids = set(venue_ids["desk_rejected"])

    target_key = "accepted" if args.decision == "accept" else "rejected"
    print(f"Target: {args.decision}")
    print(f"Candidate venue ids: {venue_ids[target_key]}")
    print(f"Desk-rejected venue ids excluded: {venue_ids['desk_rejected']}")

    print("[2] Fetching submissions...")
    notes, used_venue_ids = fetch_notes_for_venue_ids(client, venue_ids[target_key])
    filtered_notes = [
        note for note in notes
        if not is_desk_reject(note, desk_rejected_venue_ids)
    ]
    filtered_notes.sort(key=lambda note: getattr(note, "number", 0) or 0)

    if args.max_papers is not None:
        filtered_notes = filtered_notes[:args.max_papers]

    records = [paper_to_record(note) for note in tqdm(filtered_notes)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = "accepted_papers" if args.decision == "accept" else "rejected_papers"
    data_out = out_dir / f"{stem}.{args.format}"
    meta_out = out_dir / f"{stem}_crawl_meta.json"

    if args.format == "jsonl":
        write_jsonl(data_out, records)
    else:
        write_csv(data_out, records)

    meta = {
        "venue_id": VENUE_ID,
        "decision": args.decision,
        "candidate_venue_ids": venue_ids[target_key],
        "used_venue_ids": used_venue_ids,
        "desk_rejected_venue_ids_excluded": venue_ids["desk_rejected"],
        "num_papers": len(records),
        "fields": ["paper_id", "number", "title", "abstract", "primary_area", "url"],
        "downloads_pdf": False,
    }
    meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done.")
    print(f"Paper metadata output: {data_out}")
    print(f"Metadata:              {meta_out}")


if __name__ == "__main__":
    main()
