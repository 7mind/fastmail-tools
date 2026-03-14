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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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


DOWNLOAD_WORKERS = 8


def _fetch_email_batch(session: JmapSession, batch: list[str]) -> list[Email]:
    """Fetch a single batch of emails by ID."""
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

    batch_emails = []
    for raw in result[1]["list"]:
        body_values = raw.get("bodyValues", {})
        batch_emails.append(parse_email(raw, body_values))
    return batch_emails


def fetch_emails(session: JmapSession, email_ids: list[str]) -> list[Email]:
    """Fetch full email data in parallel batches."""
    batches = [
        email_ids[i : i + QUERY_BATCH_SIZE]
        for i in range(0, len(email_ids), QUERY_BATCH_SIZE)
    ]

    all_emails: list[Email] = []
    lock = threading.Lock()
    fetched_count = 0

    def fetch_and_report(batch: list[str]) -> list[Email]:
        nonlocal fetched_count
        result = _fetch_email_batch(session, batch)
        with lock:
            fetched_count += len(result)
            click.echo(f"  Fetched {fetched_count}/{len(email_ids)} emails...")
        return result

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(fetch_and_report, b): i for i, b in enumerate(batches)}
        # Collect results preserving original order
        results_by_index: dict[int, list[Email]] = {}
        for future in as_completed(futures):
            idx = futures[future]
            results_by_index[idx] = future.result()

    for i in range(len(batches)):
        all_emails.extend(results_by_index[i])

    return all_emails


def download_pdf_attachments(
    session: JmapSession,
    emails: list[Email],
) -> dict[str, bytes]:
    """Download all PDF attachments in parallel. Returns mapping blob_id -> pdf_bytes."""
    # Deduplicate by blob_id
    unique_attachments: dict[str, Attachment] = {}
    for email in emails:
        for att in email.pdf_attachments:
            if att.blob_id not in unique_attachments:
                unique_attachments[att.blob_id] = att

    if not unique_attachments:
        return {}

    total = len(unique_attachments)
    click.echo(f"Downloading {total} PDF attachment(s)...")

    pdf_blobs: dict[str, bytes] = {}
    lock = threading.Lock()
    downloaded_count = 0

    def download_one(blob_id: str, att: Attachment) -> tuple[str, bytes]:
        nonlocal downloaded_count
        data = download_blob(session, blob_id, att.name)
        with lock:
            downloaded_count += 1
            click.echo(f"  Downloaded {downloaded_count}/{total}: {att.name}")
        return blob_id, data

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = [
            pool.submit(download_one, blob_id, att)
            for blob_id, att in unique_attachments.items()
        ]
        for future in as_completed(futures):
            blob_id, data = future.result()
            pdf_blobs[blob_id] = data

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
    margin-bottom: 2em;
    padding-bottom: 1.5em;
    border-bottom: 1px solid #ccc;
  }}
  .email:last-child {{
    border-bottom: none;
  }}
  .email-header {{
    page-break-inside: avoid;
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
    """Strip <html>, <head>, <body> wrappers and trailing whitespace from email HTML."""
    html = re.sub(r"<\s*!DOCTYPE[^>]*>", "", html, flags=re.IGNORECASE)
    html = re.sub(r"<\s*/?\s*html[^>]*>", "", html, flags=re.IGNORECASE)
    html = re.sub(r"<\s*head\b[^>]*>.*?<\s*/\s*head\s*>", "", html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r"<\s*/?\s*body[^>]*>", "", html, flags=re.IGNORECASE)
    # Remove trailing empty tags, <br>, &nbsp;, and whitespace that cause blank pages
    html = re.sub(r"(<\s*br\s*/?\s*>|\s|&nbsp;|<\s*div\s*>\s*</\s*div\s*>|<\s*p\s*>\s*</\s*p\s*>)+$", "", html, flags=re.IGNORECASE)
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
            label = f"{escape_html(att.name)} [{escape_html(att.media_type)}] ({format_size(att.size)})"
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


@dataclass
class PageMeta:
    """Metadata for a single page in the final merged PDF."""
    attachment: Optional[Attachment]
    email_subject: Optional[str]


def merge_pdfs(
    base_pdf_bytes: bytes,
    emails: list[Email],
    pdf_blobs: dict[str, bytes],
) -> tuple[bytes, list[PageMeta]]:
    """Merge the rendered email PDF with any PDF attachments.

    Returns the merged PDF bytes and per-page metadata.
    """
    from pypdf import PdfReader, PdfWriter, Transformation
    from pypdf.generic import RectangleObject

    # A4 in points (1 pt = 1/72 inch)
    A4_WIDTH = 595.276
    A4_HEIGHT = 841.890

    writer = PdfWriter()
    page_metas: list[PageMeta] = []

    # Add all pages from the rendered email PDF
    base_reader = PdfReader(io.BytesIO(base_pdf_bytes))
    for page in base_reader.pages:
        writer.add_page(page)
        page_metas.append(PageMeta(attachment=None, email_subject=None))

    # Append PDF attachments at the end, grouped by email
    for email in emails:
        for att in email.pdf_attachments:
            blob = pdf_blobs.get(att.blob_id)
            if not blob:
                continue
            try:
                att_reader = PdfReader(io.BytesIO(blob))
                for page in att_reader.pages:
                    box = page.mediabox
                    src_w = float(box.width)
                    src_h = float(box.height)

                    scale_x = A4_WIDTH / src_w
                    scale_y = A4_HEIGHT / src_h
                    scale = min(scale_x, scale_y)

                    offset_x = (A4_WIDTH - src_w * scale) / 2
                    offset_y = (A4_HEIGHT - src_h * scale) / 2

                    page.add_transformation(
                        Transformation().scale(scale).translate(offset_x, offset_y)
                    )
                    page.mediabox = RectangleObject([0, 0, A4_WIDTH, A4_HEIGHT])
                    writer.add_page(page)
                    page_metas.append(PageMeta(
                        attachment=att,
                        email_subject=email.subject,
                    ))
            except Exception as exc:
                click.echo(f"  Warning: could not read PDF attachment '{att.name}' "
                           f"from email '{email.subject}': {exc}", err=True)

    output = io.BytesIO()
    writer.write(output)
    return output.getvalue(), page_metas


OVERLAY_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @page {{
    size: A4;
    margin: 0;
  }}
  body {{
    margin: 0;
    padding: 0;
  }}
  .overlay-page {{
    width: 210mm;
    height: 297mm;
    position: relative;
    page-break-after: always;
  }}
  .overlay-page:last-child {{
    page-break-after: auto;
  }}
  .header {{
    position: absolute;
    top: 0.5cm;
    left: 1.5cm;
    right: 1.5cm;
    font-family: "DejaVu Sans", "Liberation Sans", Arial, sans-serif;
    font-size: 8pt;
    color: #555;
    background: rgba(255, 255, 255, 0.85);
    padding: 0.2em 0.5em;
    border-bottom: 1px solid #ccc;
  }}
  .footer {{
    position: absolute;
    bottom: 0.5cm;
    left: 1.5cm;
    right: 1.5cm;
    font-family: "DejaVu Sans", "Liberation Sans", Arial, sans-serif;
    font-size: 8pt;
    color: #888;
    text-align: center;
  }}
</style>
</head>
<body>
{pages}
</body>
</html>
"""


def apply_overlays(
    pdf_bytes: bytes,
    page_metas: list[PageMeta],
    thread_title: str,
) -> bytes:
    """Apply footer (page numbers + thread name) and header (attachment info) overlays."""
    from pypdf import PdfReader, PdfWriter
    from weasyprint import HTML

    total_pages = len(page_metas)

    overlay_pages_html = []
    for i, meta in enumerate(page_metas):
        page_num = i + 1
        footer = escape_html(f'Thread \u201c{thread_title}\u201d, page {page_num}/{total_pages}')

        header = ""
        if meta.attachment:
            att = meta.attachment
            header_text = (
                f"Attachment: {att.name} [{att.media_type}] ({format_size(att.size)}) "
                f'\u2014 from \u201c{meta.email_subject}\u201d'
            )
            header = f'<div class="header">{escape_html(header_text)}</div>'

        overlay_pages_html.append(
            f'<div class="overlay-page">'
            f'{header}'
            f'<div class="footer">{footer}</div>'
            f'</div>'
        )

    overlay_html = OVERLAY_HTML_TEMPLATE.format(pages="\n".join(overlay_pages_html))
    overlay_pdf_bytes = HTML(string=overlay_html).write_pdf()

    overlay_reader = PdfReader(io.BytesIO(overlay_pdf_bytes))
    content_reader = PdfReader(io.BytesIO(pdf_bytes))
    writer = PdfWriter()

    for i, content_page in enumerate(content_reader.pages):
        overlay_page = overlay_reader.pages[i]
        content_page.merge_page(overlay_page)
        writer.add_page(content_page)

    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def render_single_pdf(emails: list[Email], output_path: str, pdf_blobs: dict[str, bytes]) -> None:
    emails_sorted = sorted(emails, key=lambda e: e.sent_at)
    title = f"Email Archive \u2014 {len(emails)} messages"

    base_pdf = render_emails_to_pdf_bytes(emails_sorted, title)
    merged_pdf, page_metas = merge_pdfs(base_pdf, emails_sorted, pdf_blobs)
    final_pdf = apply_overlays(merged_pdf, page_metas, title)

    with open(output_path, "wb") as f:
        f.write(final_pdf)
    click.echo(f"Written: {output_path}")


RENDER_WORKERS = os.cpu_count() or 4


def _render_one_thread(
    thread_emails: list[Email],
    pdf_blobs: dict[str, bytes],
    output_dir: str,
) -> str:
    """Render a single thread to a PDF file. Returns the output filepath."""
    thread_emails.sort(key=lambda e: e.sent_at)
    first = thread_emails[0]

    title = f"Thread: {first.subject} ({len(thread_emails)} messages)"
    base_pdf = render_emails_to_pdf_bytes(thread_emails, title)
    merged_pdf, page_metas = merge_pdfs(base_pdf, thread_emails, pdf_blobs)
    final_pdf = apply_overlays(merged_pdf, page_metas, first.subject)

    date_prefix = first.sent_at.strftime("%Y%m%d")
    filename = f"{date_prefix}_{sanitize_filename(first.subject)}.pdf"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "wb") as f:
        f.write(final_pdf)
    return filepath


def prepare_output_dir(output_dir: str) -> None:
    """Create output dir if needed, validate contents, and clean existing PDFs."""
    os.makedirs(output_dir, exist_ok=True)

    for entry in os.scandir(output_dir):
        if entry.is_dir(follow_symlinks=False):
            raise click.ClickException(
                f"Output directory '{output_dir}' contains subdirectory '{entry.name}'. "
                f"Refusing to proceed — directory must only contain .pdf files."
            )
        if not entry.name.lower().endswith(".pdf"):
            raise click.ClickException(
                f"Output directory '{output_dir}' contains non-PDF file '{entry.name}'. "
                f"Refusing to proceed — directory must only contain .pdf files."
            )

    # Clean existing PDFs
    removed = 0
    for entry in os.scandir(output_dir):
        os.remove(entry.path)
        removed += 1
    if removed:
        click.echo(f"Cleaned {removed} existing PDF(s) from {output_dir}")


def render_per_thread_pdfs(emails: list[Email], output_dir: str, pdf_blobs: dict[str, bytes]) -> None:
    prepare_output_dir(output_dir)

    threads: dict[str, list[Email]] = {}
    for email in emails:
        threads.setdefault(email.thread_id, []).append(email)

    lock = threading.Lock()
    rendered_count = 0
    total = len(threads)

    def render_and_report(thread_emails: list[Email]) -> str:
        nonlocal rendered_count
        filepath = _render_one_thread(thread_emails, pdf_blobs, output_dir)
        with lock:
            rendered_count += 1
            click.echo(f"  Written ({rendered_count}/{total}): {filepath}")
        return filepath

    with ThreadPoolExecutor(max_workers=RENDER_WORKERS) as pool:
        futures = [
            pool.submit(render_and_report, thread_emails)
            for thread_emails in threads.values()
        ]
        for future in as_completed(futures):
            future.result()

    click.echo(f"Created {total} thread PDFs in {output_dir}")


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
