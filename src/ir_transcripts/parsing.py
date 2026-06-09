from __future__ import annotations

from io import BytesIO
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from pypdf import PdfReader

from .models import CandidateLink


def visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    return " ".join(soup.get_text(" ").split())


def page_title(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    if soup.title and soup.title.string:
        return " ".join(soup.title.string.split())
    heading = soup.find(["h1", "h2"])
    return " ".join(heading.get_text(" ").split()) if heading else "Untitled"


def extract_links(html: str, source_url: str) -> list[CandidateLink]:
    soup = BeautifulSoup(html, "lxml")
    links: list[CandidateLink] = []
    for anchor in soup.find_all("a", href=True):
        label = " ".join(anchor.get_text(" ").split())
        absolute = urljoin(source_url, anchor["href"])
        parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"}:
            continue
        links.append(CandidateLink(url=absolute, label=label[:250], source_url=source_url))
    return links


def pdf_text(content: bytes) -> str:
    reader = PdfReader(BytesIO(content))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n\n".join(part for part in parts if part.strip())


def looks_like_transcript(text: str) -> bool:
    lower = text.lower()
    markers = [
        "earnings call transcript",
        "question-and-answer session",
        "conference call",
        "prepared remarks",
        "operator",
        "analysts",
    ]
    return sum(marker in lower for marker in markers) >= 2


def looks_like_js_shell(html: str, text: str) -> bool:
    lower = html.lower()
    if len(text) < 500 and len(html) > 5000:
        return True
    markers = ("__next_data__", "data-reactroot", "id=\"root\"", "id=\"app\"", "window.__")
    return len(text) < 1500 and any(marker in lower for marker in markers)
