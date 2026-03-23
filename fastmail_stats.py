#!/usr/bin/env python3
"""Fastmail inbox statistics tool.

Connects to Fastmail via JMAP API, scans the Inbox, and displays
histograms of sender domains and sender email addresses sorted by frequency.
"""

import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import click

from fastmail_downloader import (
    DOWNLOAD_WORKERS,
    JmapSession,
    QUERY_BATCH_SIZE,
    create_session,
    jmap_call,
)


@dataclass
class SenderInfo:
    email: str
    domain: str


def get_inbox_id(session: JmapSession) -> str:
    """Find the Inbox mailbox ID."""
    responses = jmap_call(session, [
        [
            "Mailbox/get",
            {
                "accountId": session.account_id,
                "properties": ["id", "role"],
            },
            "m0",
        ]
    ])

    result = responses[0]
    assert result[0] == "Mailbox/get", f"Unexpected response: {result[0]}"

    for mailbox in result[1]["list"]:
        if mailbox.get("role") == "inbox":
            return mailbox["id"]

    raise AssertionError("No Inbox mailbox found")


def query_inbox_email_ids(session: JmapSession, inbox_id: str) -> list[str]:
    """Query all email IDs in the Inbox using pagination."""
    all_ids: list[str] = []
    position = 0

    while True:
        responses = jmap_call(session, [
            [
                "Email/query",
                {
                    "accountId": session.account_id,
                    "filter": {"inMailbox": inbox_id},
                    "sort": [{"property": "receivedAt", "isAscending": False}],
                    "position": position,
                    "limit": QUERY_BATCH_SIZE,
                },
                "q0",
            ]
        ])

        result = responses[0]
        assert result[0] == "Email/query", f"Unexpected response: {result[0]}"
        ids = result[1]["ids"]
        all_ids.extend(ids)

        click.echo(f"  Queried {len(all_ids)} email IDs so far...")

        if len(ids) < QUERY_BATCH_SIZE:
            break

        position += len(ids)

    return all_ids


def _fetch_senders_batch(session: JmapSession, batch: list[str]) -> list[SenderInfo]:
    """Fetch sender addresses for a batch of email IDs."""
    responses = jmap_call(session, [
        [
            "Email/get",
            {
                "accountId": session.account_id,
                "ids": batch,
                "properties": ["from"],
            },
            "g0",
        ]
    ])

    result = responses[0]
    assert result[0] == "Email/get", f"Unexpected response: {result[0]}"

    senders: list[SenderInfo] = []
    for raw in result[1]["list"]:
        from_list = raw.get("from") or []
        for addr in from_list:
            email = addr.get("email", "")
            if email:
                domain = email.split("@", 1)[1] if "@" in email else email
                senders.append(SenderInfo(email=email.lower(), domain=domain.lower()))
    return senders


def fetch_all_senders(session: JmapSession, email_ids: list[str]) -> list[SenderInfo]:
    """Fetch sender info for all emails in parallel batches."""
    batches = [
        email_ids[i : i + QUERY_BATCH_SIZE]
        for i in range(0, len(email_ids), QUERY_BATCH_SIZE)
    ]

    all_senders: list[SenderInfo] = []
    lock = threading.Lock()
    fetched_count = 0

    def fetch_and_report(batch: list[str]) -> list[SenderInfo]:
        nonlocal fetched_count
        result = _fetch_senders_batch(session, batch)
        with lock:
            fetched_count += len(batch)
            click.echo(f"  Fetched senders for {fetched_count}/{len(email_ids)} emails...")
        return result

    results_by_index: dict[int, list[SenderInfo]] = {}
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(fetch_and_report, b): i for i, b in enumerate(batches)}
        for future in as_completed(futures):
            idx = futures[future]
            results_by_index[idx] = future.result()

    for i in range(len(batches)):
        all_senders.extend(results_by_index[i])

    return all_senders


def print_histogram(title: str, counter: Counter[str], top_n: int) -> None:
    """Print a formatted histogram table."""
    entries = counter.most_common(top_n)
    if not entries:
        click.echo(f"\n{title}: (no data)")
        return

    max_key_len = max(len(entry[0]) for entry in entries)
    max_count_len = max(len(str(entry[1])) for entry in entries)
    total = sum(counter.values())

    click.echo(f"\n{'=' * 70}")
    click.echo(f" {title} (showing top {min(top_n, len(entries))} of {len(counter)} unique)")
    click.echo(f"{'=' * 70}")
    click.echo(f" {'#':<5} {'Count':>{max_count_len}}  {'%':>6}  {'Name'}")
    click.echo(f" {'-' * 4} {'-' * max_count_len}  {'-' * 6}  {'-' * max_key_len}")

    for rank, (key, count) in enumerate(entries, 1):
        pct = count / total * 100
        click.echo(f" {rank:<5} {count:>{max_count_len}}  {pct:>5.1f}%  {key}")

    click.echo(f" {'-' * 4} {'-' * max_count_len}  {'-' * 6}  {'-' * max_key_len}")
    click.echo(f" Total: {total} emails from {len(counter)} unique entries")


@click.command()
@click.option(
    "--token",
    envvar="FASTMAIL_API_TOKEN",
    required=True,
    help="Fastmail API token (or set FASTMAIL_API_TOKEN env var).",
)
@click.option(
    "--top-n",
    type=int,
    default=50,
    help="Number of top entries to show in each histogram. Default: 50.",
)
def main(token: str, top_n: int) -> None:
    """Show sender statistics for Fastmail Inbox."""
    assert top_n > 0, "--top-n must be a positive integer"

    click.echo("Connecting to Fastmail...")
    session = create_session(token)
    click.echo(f"Connected. Account: {session.account_id}")

    click.echo("Finding Inbox...")
    inbox_id = get_inbox_id(session)
    click.echo(f"Inbox ID: {inbox_id}")

    click.echo("Querying Inbox emails...")
    email_ids = query_inbox_email_ids(session, inbox_id)
    click.echo(f"Found {len(email_ids)} emails in Inbox.")

    if not email_ids:
        click.echo("No emails in Inbox.")
        return

    click.echo("Fetching sender information...")
    senders = fetch_all_senders(session, email_ids)
    click.echo(f"Collected {len(senders)} sender addresses.")

    domain_counts: Counter[str] = Counter()
    email_counts: Counter[str] = Counter()
    for sender in senders:
        domain_counts[sender.domain] += 1
        email_counts[sender.email] += 1

    print_histogram("Top Sender Domains", domain_counts, top_n)
    print_histogram("Top Sender Emails", email_counts, top_n)


if __name__ == "__main__":
    main()
