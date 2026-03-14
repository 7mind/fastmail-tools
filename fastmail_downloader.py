#!/usr/bin/env python3
"""Fastmail email downloader and PDF exporter.

Connects to Fastmail via JMAP API, downloads emails matching address filters,
and renders them to PDF — either as a single chronological file or one file per thread.
PDF attachments are downloaded and appended to the rendered output.
"""

import io
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import click
import requests
from dateutil.parser import isoparse


# --- JMAP Client ---

FASTMAIL_SESSION_URL = "https://api.fastmail.com/.well-known/jmap"


@dataclass
class JmapSession:
    headers: dict[str, str]
    account_id: str
    api_url: str
    download_url: str


def create_session(api_token: str) -> JmapSession:
    headers = {"Authorization": f"Bearer {api_token}"}
    response = requests.get(FASTMAIL_SESSION_URL, headers=headers)
    response.raise_for_status()
    data = response.json()

    accounts = data["accounts"]
    assert len(accounts) > 0, "No accounts found in JMAP session"
    account_id = data["primaryAccounts"]["urn:ietf:params:jmap:mail"]

    return JmapSession(
        headers=headers,
        account_id=account_id,
        api_url=data["apiUrl"],
        download_url=data["downloadUrl"],
    )


def jmap_call(session: JmapSession, method_calls: list[list]) -> list:
    payload = {
        "using": ["urn:ietf:params:jmap:core", "urn:ietf:params:jmap:mail"],
        "methodCalls": method_calls,
    }
    response = requests.post(
        session.api_url,
        headers={**session.headers, "Content-Type": "application/json"},
        json=payload,
    )
    response.raise_for_status()
    return response.json()["methodResponses"]


def download_blob(session: JmapSession, blob_id: str, name: str) -> bytes:
    """Download a blob (attachment) from Fastmail."""
    url = session.download_url.replace(
        "{accountId}", session.account_id
    ).replace(
        "{blobId}", blob_id
    ).replace(
        "{type}", "application/octet-stream"
    ).replace(
        "{name}", name
    )
    response = requests.get(url, headers=session.headers)
    response.raise_for_status()
    return response.content


# --- Email Data Model ---

@dataclass
class EmailAddress:
    name: Optional[str]
    email: str

    def __str__(self) -> str:
        if self.name:
            return f"{self.name} <{self.email}>"
        return self.email


@dataclass
class Attachment:
    name: str
    media_type: str
    size: int
    blob_id: str

    @property
    def is_pdf(self) -> bool:
        return self.media_type == "application/pdf"


@dataclass
class Email:
    id: str
    thread_id: str
    subject: str
    sent_at: datetime
    received_at: datetime
    from_addresses: list[EmailAddress]
    to_addresses: list[EmailAddress]
    cc_addresses: list[EmailAddress]
    bcc_addresses: list[EmailAddress]
    text_body: str
    html_body: str
    preview: str
    attachments: list[Attachment]

    @property
    def from_str(self) -> str:
        return ", ".join(str(a) for a in self.from_addresses)

    @property
    def to_str(self) -> str:
        return ", ".join(str(a) for a in self.to_addresses)

    @property
    def cc_str(self) -> str:
        return ", ".join(str(a) for a in self.cc_addresses)

    @property
    def bcc_str(self) -> str:
        return ", ".join(str(a) for a in self.bcc_addresses)

    @property
    def pdf_attachments(self) -> list[Attachment]:
        return [a for a in self.attachments if a.is_pdf]


def parse_addresses(addr_list: Optional[list[dict]]) -> list[EmailAddress]:
    if not addr_list:
        return []
    return [
        EmailAddress(name=a.get("name"), email=a["email"])
        for a in addr_list
    ]


def parse_attachments(att_list: Optional[list[dict]]) -> list[Attachment]:
    if not att_list:
        return []
    return [
        Attachment(
            name=a.get("name") or "unnamed",
            media_type=a.get("type") or "application/octet-stream",
            size=a.get("size") or 0,
            blob_id=a["blobId"],
        )
        for a in att_list
    ]


def parse_email(raw: dict, body_values: dict) -> Email:
    text_body = ""
    html_body = ""

    for part in raw.get("textBody", []):
        part_id = part["partId"]
        if part_id in body_values:
            text_body += body_values[part_id].get("value", "")

    for part in raw.get("htmlBody", []):
        part_id = part["partId"]
        if part_id in body_values:
            html_body += body_values[part_id].get("value", "")

    sent_at_str = raw.get("sentAt")
    received_at_str = raw["receivedAt"]

    sent_at = isoparse(sent_at_str) if sent_at_str else isoparse(received_at_str)
    received_at = isoparse(received_at_str)

    return Email(
        id=raw["id"],
        thread_id=raw["threadId"],
        subject=raw.get("subject", "(no subject)"),
        sent_at=sent_at,
        received_at=received_at,
        from_addresses=parse_addresses(raw.get("from")),
        to_addresses=parse_addresses(raw.get("to")),
        cc_addresses=parse_addresses(raw.get("cc")),
        bcc_addresses=parse_addresses(raw.get("bcc")),
        text_body=text_body,
        html_body=html_body,
        preview=raw.get("preview", ""),
        attachments=parse_attachments(raw.get("attachments")),
    )


# --- JMAP Email Querying ---

QUERY_BATCH_SIZE = 50


def build_address_filter(
    from_list: list[str],
    to_list: list[str],
    cc_list: list[str],
) -> Optional[dict]:
    """Build a JMAP filter matching any of the given address substrings.

    Each address list produces OR conditions for that field.
    The resulting filter matches emails where ANY of the from_list matches the From field,
    OR ANY of the to_list matches the To field, OR ANY of the cc_list matches the Cc field.
    """
    conditions = []

    for substring in from_list:
        conditions.append({"from": substring})
    for substring in to_list:
        conditions.append({"to": substring})
    for substring in cc_list:
        conditions.append({"cc": substring})

    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {"operator": "OR", "conditions": conditions}


def build_filter(
    range_start: Optional[datetime],
    range_end: Optional[datetime],
    from_list: list[str],
    to_list: list[str],
    cc_list: list[str],
) -> dict:
    parts = []

    if range_start:
        parts.append({"after": range_start.strftime("%Y-%m-%dT%H:%M:%SZ")})
    if range_end:
        parts.append({"before": range_end.strftime("%Y-%m-%dT%H:%M:%SZ")})

    address_filter = build_address_filter(from_list, to_list, cc_list)
    if address_filter:
        parts.append(address_filter)

    assert len(parts) > 0, "At least one filter criterion is required"

    if len(parts) == 1:
        return parts[0]

    return {"operator": "AND", "conditions": parts}


def query_email_ids(session: JmapSession, email_filter: dict) -> list[str]:
    """Query all matching email IDs using pagination."""
    all_ids: list[str] = []
    position = 0

    while True:
        responses = jmap_call(session, [
            [
                "Email/query",
                {
                    "accountId": session.account_id,
                    "filter": email_filter,
                    "sort": [{"property": "receivedAt", "isAscending": True}],
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


def fetch_emails(session: JmapSession, email_ids: list[str]) -> list[Email]:
    """Fetch full email data in batches."""
    emails: list[Email] = []

    for i in range(0, len(email_ids), QUERY_BATCH_SIZE):
        batch = email_ids[i : i + QUERY_BATCH_SIZE]

        responses = jmap_call(session, [
            [
                "Email/get",
                {
                    "accountId": session.account_id,
                    "ids": batch,
                    "properties": [
                        "id", "threadId", "subject", "sentAt", "receivedAt",
                        "from", "to", "cc", "bcc", "preview",
                        "textBody", "htmlBody", "bodyValues",
                        "attachments",
                    ],
                    "fetchHTMLBodyValues": True,
                    "fetchTextBodyValues": True,
                },
                "g0",
            ]
        ])

        result = responses[0]
        assert result[0] == "Email/get", f"Unexpected response: {result[0]}"

        for raw in result[1]["list"]:
            body_values = raw.get("bodyValues", {})
            emails.append(parse_email(raw, body_values))

        click.echo(f"  Fetched {len(emails)}/{len(email_ids)} emails...")

    return emails


def download_pdf_attachments(
    session: JmapSession,
    emails: list[Email],
) -> dict[str, bytes]:
    """Download all PDF attachments. Returns mapping blob_id -> pdf_bytes."""
    pdf_blobs: dict[str, bytes] = {}
    total = sum(len(e.pdf_attachments) for e in emails)
    if total == 0:
        return pdf_blobs

    click.echo(f"Downloading {total} PDF attachment(s)...")
    count = 0
    for email in emails:
        for att in email.pdf_attachments:
            if att.blob_id not in pdf_blobs:
                pdf_blobs[att.blob_id] = download_blob(session, att.blob_id, att.name)
                count += 1
                click.echo(f"  Downloaded {count}/{total}: {att.name}")

    return pdf_blobs


# --- PDF Rendering ---

EMAIL_HTML_TEMPLATE = """\
<div class="email">
  <div class="email-header">
    <div class="email-field"><span class="label">Date:</span> {sent_at}</div>
    <div class="email-field"><span class="label">From:</span> {from_str}</div>
    <div class="email-field"><span class="label">To:</span> {to_str}</div>
    {cc_line}
    {bcc_line}
    <div class="email-field"><span class="label">Subject:</span> {subject}</div>
    {attachments_line}
  </div>
  <div class="email-body">
    {body}
  </div>
</div>
"""

PAGE_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @page {{
    size: A4;
    margin: 2cm;
  }}
  body {{
    font-family: "DejaVu Sans", "Liberation Sans", Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.4;
    color: #1a1a1a;
  }}
  .email {{
    page-break-inside: avoid;
    margin-bottom: 2em;
    padding-bottom: 1.5em;
    border-bottom: 1px solid #ccc;
  }}
  .email:last-child {{
    border-bottom: none;
  }}
  .email-header {{
    background: #f5f5f5;
    padding: 0.8em;
    margin-bottom: 1em;
    border-left: 3px solid #4a90d9;
    font-size: 10pt;
  }}
  .email-field {{
    margin-bottom: 0.2em;
  }}
  .label {{
    font-weight: bold;
    color: #555;
  }}
  .attachment-list {{
    margin-top: 0.3em;
  }}
  .attachment-item {{
    display: inline-block;
    background: #e8e8e8;
    padding: 0.1em 0.5em;
    margin: 0.1em 0.3em 0.1em 0;
    border-radius: 3px;
    font-size: 9pt;
  }}
  .attachment-item.pdf {{
    background: #fde8e8;
    border: 1px solid #e0a0a0;
  }}
  .email-body {{
    padding: 0.5em;
  }}
  .email-body img {{
    max-width: 100%;
  }}
  .email-body table {{
    max-width: 100%;
  }}
  .text-body {{
    white-space: pre-wrap;
    font-family: "DejaVu Sans Mono", "Liberation Mono", monospace;
    font-size: 10pt;
  }}
  h1 {{
    font-size: 16pt;
    color: #333;
    border-bottom: 2px solid #4a90d9;
    padding-bottom: 0.3em;
  }}
  .thread-separator {{
    page-break-before: always;
  }}
</style>
</head>
<body>
<h1>{title}</h1>
{content}
</body>
</html>
"""


def sanitize_html_body(html: str) -> str:
    """Strip <html>, <head>, <body> wrappers from email HTML to embed safely."""
    html = re.sub(r"<\s*!DOCTYPE[^>]*>", "", html, flags=re.IGNORECASE)
    html = re.sub(r"<\s*/?\s*html[^>]*>", "", html, flags=re.IGNORECASE)
    html = re.sub(r"<\s*head\b[^>]*>.*?<\s*/\s*head\s*>", "", html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r"<\s*/?\s*body[^>]*>", "", html, flags=re.IGNORECASE)
    return html


def escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def render_email_html(email: Email) -> str:
    cc_line = ""
    if email.cc_addresses:
        cc_line = f'<div class="email-field"><span class="label">Cc:</span> {escape_html(email.cc_str)}</div>'

    bcc_line = ""
    if email.bcc_addresses:
        bcc_line = f'<div class="email-field"><span class="label">Bcc:</span> {escape_html(email.bcc_str)}</div>'

    attachments_line = ""
    if email.attachments:
        items = []
        for att in email.attachments:
            css_class = "attachment-item pdf" if att.is_pdf else "attachment-item"
            label = f"{escape_html(att.name)} ({format_size(att.size)})"
            items.append(f'<span class="{css_class}">{label}</span>')
        attachments_line = (
            f'<div class="email-field"><span class="label">Attachments:</span>'
            f'<div class="attachment-list">{"".join(items)}</div></div>'
        )

    if email.html_body.strip():
        body = sanitize_html_body(email.html_body)
    elif email.text_body.strip():
        body = f'<div class="text-body">{escape_html(email.text_body)}</div>'
    else:
        body = f'<div class="text-body">{escape_html(email.preview)}</div>'

    return EMAIL_HTML_TEMPLATE.format(
        sent_at=escape_html(email.sent_at.strftime("%Y-%m-%d %H:%M:%S %Z")),
        from_str=escape_html(email.from_str),
        to_str=escape_html(email.to_str),
        cc_line=cc_line,
        bcc_line=bcc_line,
        subject=escape_html(email.subject),
        attachments_line=attachments_line,
        body=body,
    )


def sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", "_", name)
    name = name.strip("_.")
    return name[:200] if name else "untitled"


def render_emails_to_pdf_bytes(emails_sorted: list[Email], title: str) -> bytes:
    """Render a list of emails to PDF bytes (without attachment merging)."""
    from weasyprint import HTML

    content = "\n".join(render_email_html(e) for e in emails_sorted)
    full_html = PAGE_HTML_TEMPLATE.format(title=escape_html(title), content=content)
    return HTML(string=full_html).write_pdf()


def merge_pdfs(base_pdf_bytes: bytes, emails: list[Email], pdf_blobs: dict[str, bytes]) -> bytes:
    """Merge the rendered email PDF with any PDF attachments.

    For each email that has PDF attachments, the attachment pages are appended
    after the email content in the final PDF.
    """
    from pypdf import PdfReader, PdfWriter

    has_any_pdfs = any(e.pdf_attachments for e in emails)
    if not has_any_pdfs:
        return base_pdf_bytes

    writer = PdfWriter()

    # Add all pages from the rendered email PDF
    base_reader = PdfReader(io.BytesIO(base_pdf_bytes))
    for page in base_reader.pages:
        writer.add_page(page)

    # Append PDF attachments at the end, grouped by email
    for email in emails:
        for att in email.pdf_attachments:
            blob = pdf_blobs.get(att.blob_id)
            if not blob:
                continue
            try:
                att_reader = PdfReader(io.BytesIO(blob))
                for page in att_reader.pages:
                    writer.add_page(page)
            except Exception as exc:
                click.echo(f"  Warning: could not read PDF attachment '{att.name}' "
                           f"from email '{email.subject}': {exc}", err=True)

    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def render_single_pdf(emails: list[Email], output_path: str, pdf_blobs: dict[str, bytes]) -> None:
    emails_sorted = sorted(emails, key=lambda e: e.sent_at)
    title = f"Email Archive \u2014 {len(emails)} messages"

    base_pdf = render_emails_to_pdf_bytes(emails_sorted, title)
    final_pdf = merge_pdfs(base_pdf, emails_sorted, pdf_blobs)

    with open(output_path, "wb") as f:
        f.write(final_pdf)
    click.echo(f"Written: {output_path}")


def render_per_thread_pdfs(emails: list[Email], output_dir: str, pdf_blobs: dict[str, bytes]) -> None:
    os.makedirs(output_dir, exist_ok=True)

    threads: dict[str, list[Email]] = {}
    for email in emails:
        threads.setdefault(email.thread_id, []).append(email)

    for thread_id, thread_emails in threads.items():
        thread_emails.sort(key=lambda e: e.sent_at)
        first = thread_emails[0]

        title = f"Thread: {first.subject} ({len(thread_emails)} messages)"
        base_pdf = render_emails_to_pdf_bytes(thread_emails, title)
        final_pdf = merge_pdfs(base_pdf, thread_emails, pdf_blobs)

        date_prefix = first.sent_at.strftime("%Y%m%d")
        filename = f"{date_prefix}_{sanitize_filename(first.subject)}.pdf"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "wb") as f:
            f.write(final_pdf)
        click.echo(f"Written: {filepath}")

    click.echo(f"Created {len(threads)} thread PDFs in {output_dir}")


# --- CLI ---

@click.command()
@click.option(
    "--token",
    envvar="FASTMAIL_API_TOKEN",
    required=True,
    help="Fastmail API token (or set FASTMAIL_API_TOKEN env var).",
)
@click.option(
    "--range-start",
    type=click.DateTime(),
    default=None,
    help="Start of date range (inclusive). ISO 8601 format.",
)
@click.option(
    "--range-end",
    type=click.DateTime(),
    default=None,
    help="End of date range (exclusive). ISO 8601 format.",
)
@click.option(
    "--from-list",
    default="",
    help="Comma-separated substrings to match in From field.",
)
@click.option(
    "--to-list",
    default="",
    help="Comma-separated substrings to match in To field.",
)
@click.option(
    "--cc-list",
    default="",
    help="Comma-separated substrings to match in Cc field.",
)
@click.option(
    "--single-file",
    is_flag=True,
    default=False,
    help="Render all emails into one PDF file chronologically.",
)
@click.option(
    "--file-per-chain",
    is_flag=True,
    default=False,
    help="Render one PDF file per email thread/chain.",
)
@click.option(
    "--output",
    default="output",
    help="Output file path (for --single-file) or directory (for --file-per-chain).",
)
def main(
    token: str,
    range_start: Optional[datetime],
    range_end: Optional[datetime],
    from_list: str,
    to_list: str,
    cc_list: str,
    single_file: bool,
    file_per_chain: bool,
    output: str,
) -> None:
    """Download emails from Fastmail and export them as PDF."""

    assert single_file or file_per_chain, (
        "Specify at least one output mode: --single-file or --file-per-chain"
    )
    assert not (single_file and file_per_chain), (
        "Specify only one output mode: --single-file or --file-per-chain"
    )

    from_substrings = [s.strip() for s in from_list.split(",") if s.strip()]
    to_substrings = [s.strip() for s in to_list.split(",") if s.strip()]
    cc_substrings = [s.strip() for s in cc_list.split(",") if s.strip()]

    assert from_substrings or to_substrings or cc_substrings, (
        "At least one of --from-list, --to-list, or --cc-list must be specified"
    )

    if range_start and range_start.tzinfo is None:
        range_start = range_start.replace(tzinfo=timezone.utc)
    if range_end and range_end.tzinfo is None:
        range_end = range_end.replace(tzinfo=timezone.utc)

    click.echo("Connecting to Fastmail...")
    session = create_session(token)
    click.echo(f"Connected. Account: {session.account_id}")

    click.echo("Building query filter...")
    email_filter = build_filter(range_start, range_end, from_substrings, to_substrings, cc_substrings)
    click.echo(f"Filter: {json.dumps(email_filter, indent=2)}")

    click.echo("Querying emails...")
    email_ids = query_email_ids(session, email_filter)
    click.echo(f"Found {len(email_ids)} matching emails.")

    if not email_ids:
        click.echo("No emails to export.")
        return

    click.echo("Fetching email content...")
    emails = fetch_emails(session, email_ids)
    click.echo(f"Fetched {len(emails)} emails.")

    pdf_blobs = download_pdf_attachments(session, emails)

    click.echo("Rendering PDFs...")
    if single_file:
        if not output.endswith(".pdf"):
            output = output + ".pdf"
        render_single_pdf(emails, output, pdf_blobs)
    else:
        render_per_thread_pdfs(emails, output, pdf_blobs)

    click.echo("Done.")


if __name__ == "__main__":
    main()
