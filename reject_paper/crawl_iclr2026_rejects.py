#!/usr/bin/env python3

import argparse
import json
import os
import re
import time
from pathlib import Path

import openreview
from tqdm import tqdm


VENUE_ID = "ICLR.cc/2026/Conference"
REJECTED_VENUE_ID = f"{VENUE_ID}/Rejected_Submission"
DESK_REJECTED_VENUE_ID = f"{VENUE_ID}/Desk_Rejected_Submission"


SUBMISSION_FIELDS = [
    "title",
    "abstract",
    "TLDR",
    "keywords",
    "primary_area",
    "venue",
    "venueid",
    "paperhash",
]


def get_value(content, key, default=None):
    if not content:
        return default

    value = content.get(key, default)
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def flatten_content(content):
    flat = {}
    for key, value in (content or {}).items():
        if key == "pdf":
            continue
        if isinstance(value, dict) and "value" in value:
            flat[key] = value["value"]
        else:
            flat[key] = value
    return flat


def make_client(username=None, password=None):
    kwargs = {"baseurl": "https://api2.openreview.net"}
    if username and password:
        kwargs.update({"username": username, "password": password})
    return openreview.api.OpenReviewClient(**kwargs)


def discover_rejected_venue_id(client):
    group = client.get_group(VENUE_ID)
    content = group.content or {}
    rejected = get_value(content, "rejected_venue_id", REJECTED_VENUE_ID)
    desk_rejected = get_value(
        content,
        "desk_rejected_venue_id",
        DESK_REJECTED_VENUE_ID,
    )
    return rejected, desk_rejected


def fetch_rejected_submissions(client, rejected_venue_id, desk_rejected_venue_id):
    submissions = client.get_all_notes(content={"venueid": rejected_venue_id})
    filtered = []
    skipped_desk_rejects = 0

    for submission in submissions:
        venueid = get_value(submission.content, "venueid", "")
        venue = str(get_value(submission.content, "venue", "") or "")
        combined = f"{venueid} {venue}".lower()

        if venueid == desk_rejected_venue_id or "desk rejected" in combined:
            skipped_desk_rejects += 1
            continue
        if venueid != rejected_venue_id:
            continue
        filtered.append(submission)

    return filtered, skipped_desk_rejects


def get_signature(reply):
    signatures = getattr(reply, "signatures", None) or []
    return signatures[0] if signatures else ""


def reviewer_id(signature):
    match = re.search(r"(Reviewer_[^/]+)", signature)
    return match.group(1) if match else ""


def classify_reply(reply):
    signature = get_signature(reply)
    content = flatten_content(reply.content)
    keys = set(content.keys())
    title = str(content.get("title", "") or "").lower()

    if "decision" in keys or title == "paper decision":
        return "decision"

    if "/Reviewer_" in signature and (
        "rating" in keys
        or "confidence" in keys
        or {"summary", "strengths", "weaknesses"}.intersection(keys)
    ):
        return "official_review"

    if "/Area_Chair_" in signature or "reviewer_scores" in keys or "reviewer_concerns" in keys:
        return "area_chair_summary"

    if "comment" in keys:
        if "/Reviewer_" in signature:
            return "reviewer_comment"
        if "/Area_Chair_" in signature:
            return "area_chair_comment"
        return "public_comment"

    if "/Reviewer_" in signature:
        return "reviewer_reply"

    return "other"


def reply_to_record(reply, submission):
    signature = get_signature(reply)
    return {
        "submission_id": submission.id,
        "submission_number": getattr(submission, "number", None),
        "submission_title": get_value(submission.content, "title", ""),
        "reply_id": reply.id,
        "reply_type": classify_reply(reply),
        "reviewer_id": reviewer_id(signature),
        "forum": getattr(reply, "forum", None),
        "replyto": getattr(reply, "replyto", None),
        "signatures": getattr(reply, "signatures", None),
        "readers": getattr(reply, "readers", None),
        "cdate": getattr(reply, "cdate", None),
        "mdate": getattr(reply, "mdate", None),
        "content": flatten_content(reply.content),
    }


def fetch_replies(client, forum_id, retries=3, sleep=1.0):
    for attempt in range(1, retries + 1):
        try:
            notes = client.get_all_notes(forum=forum_id)
            return [note for note in notes if getattr(note, "replyto", None)], None
        except Exception as exc:
            if attempt == retries:
                return [], str(exc)
            time.sleep(sleep * attempt)
    return [], "unknown error"


def submission_to_record(submission, replies, reply_error=None):
    reply_records = [reply_to_record(reply, submission) for reply in replies]

    grouped = {
        "official_reviews": [],
        "reviewer_comments": [],
        "area_chair_summaries": [],
        "area_chair_comments": [],
        "decisions": [],
        "public_comments": [],
        "other_replies": [],
    }

    for reply in reply_records:
        reply_type = reply["reply_type"]
        if reply_type == "official_review":
            grouped["official_reviews"].append(reply)
        elif reply_type == "reviewer_comment":
            grouped["reviewer_comments"].append(reply)
        elif reply_type == "area_chair_summary":
            grouped["area_chair_summaries"].append(reply)
        elif reply_type == "area_chair_comment":
            grouped["area_chair_comments"].append(reply)
        elif reply_type == "decision":
            grouped["decisions"].append(reply)
        elif reply_type == "public_comment":
            grouped["public_comments"].append(reply)
        else:
            grouped["other_replies"].append(reply)

    record = {
        "paper_id": submission.id,
        "forum_id": getattr(submission, "forum", None),
        "number": getattr(submission, "number", None),
        "url": f"https://openreview.net/forum?id={submission.id}",
        "reply_error": reply_error,
    }

    for field in SUBMISSION_FIELDS:
        record[field] = get_value(submission.content, field, "")

    record.update({
        "num_replies": len(reply_records),
        "num_official_reviews": len(grouped["official_reviews"]),
        "num_reviewer_comments": len(grouped["reviewer_comments"]),
        "num_area_chair_summaries": len(grouped["area_chair_summaries"]),
        "num_area_chair_comments": len(grouped["area_chair_comments"]),
        "num_decisions": len(grouped["decisions"]),
        "num_public_comments": len(grouped["public_comments"]),
        "num_other_replies": len(grouped["other_replies"]),
        **grouped,
    })
    return record


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Crawl public OpenReview comments/reviews for ICLR 2026 rejected "
            "submissions, excluding desk rejects and never downloading PDFs."
        )
    )
    parser.add_argument("--out", default="iclr2026_reject_reviews")
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    username = os.getenv("OPENREVIEW_USERNAME")
    password = os.getenv("OPENREVIEW_PASSWORD")
    client = make_client(username, password)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    paper_out = out_dir / "rejected_papers_with_reviews.jsonl"
    flat_reply_out = out_dir / "all_replies_flat.jsonl"
    error_out = out_dir / "errors.jsonl"
    meta_out = out_dir / "crawl_meta.json"

    print("[1] Discovering ICLR 2026 venue ids...")
    rejected_venue_id, desk_rejected_venue_id = discover_rejected_venue_id(client)
    print(f"Rejected venue id:      {rejected_venue_id}")
    print(f"Desk-rejected venue id: {desk_rejected_venue_id}")

    print("[2] Fetching rejected submissions, excluding desk rejects...")
    submissions, skipped_desk_rejects = fetch_rejected_submissions(
        client,
        rejected_venue_id,
        desk_rejected_venue_id,
    )
    submissions.sort(key=lambda note: getattr(note, "number", 0) or 0)

    if args.max_papers is not None:
        submissions = submissions[:args.max_papers]

    print(f"Rejected submissions to crawl: {len(submissions)}")
    print(f"Desk rejects skipped:          {skipped_desk_rejects}")

    meta = {
        "venue_id": VENUE_ID,
        "rejected_venue_id": rejected_venue_id,
        "desk_rejected_venue_id": desk_rejected_venue_id,
        "num_submissions_to_crawl": len(submissions),
        "skipped_desk_rejects": skipped_desk_rejects,
        "downloads_pdf": False,
    }
    meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    with paper_out.open("w", encoding="utf-8") as paper_file, \
            flat_reply_out.open("w", encoding="utf-8") as reply_file, \
            error_out.open("w", encoding="utf-8") as error_file:
        for submission in tqdm(submissions):
            replies, reply_error = fetch_replies(
                client,
                submission.id,
                retries=args.retries,
                sleep=max(args.sleep, 0.5),
            )

            paper_record = submission_to_record(
                submission=submission,
                replies=replies,
                reply_error=reply_error,
            )
            paper_file.write(json.dumps(paper_record, ensure_ascii=False) + "\n")

            for reply in replies:
                reply_record = reply_to_record(reply, submission)
                reply_file.write(json.dumps(reply_record, ensure_ascii=False) + "\n")

            if reply_error:
                error_file.write(json.dumps({
                    "paper_id": submission.id,
                    "number": getattr(submission, "number", None),
                    "title": get_value(submission.content, "title", ""),
                    "reply_error": reply_error,
                }, ensure_ascii=False) + "\n")

            if args.sleep > 0:
                time.sleep(args.sleep)

    print("Done.")
    print(f"Paper-level output: {paper_out}")
    print(f"Flat reply output:  {flat_reply_out}")
    print(f"Errors:             {error_out}")
    print(f"Metadata:           {meta_out}")


if __name__ == "__main__":
    main()
