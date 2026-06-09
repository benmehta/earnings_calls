from __future__ import annotations

from io import BytesIO
from urllib.parse import urljoin, urlparse
from zipfile import ZipFile
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
from pypdf import PdfReader

from .models import CandidateLink
from .urls import resolve_document_url


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
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        link = _candidate_from_tag(anchor, "href", source_url)
        if link and link.url not in seen:
            seen.add(link.url)
            links.append(link)

    for tag in soup.find_all(attrs={"link": True}):
        link = _candidate_from_tag(tag, "link", source_url)
        if link and link.url not in seen:
            seen.add(link.url)
            links.append(link)
    return links


def _candidate_from_tag(tag, attribute: str, source_url: str) -> CandidateLink | None:
    absolute = resolve_document_url(urljoin(source_url, tag.get(attribute, "")))
    parsed = urlparse(absolute)
    if parsed.scheme not in {"http", "https"}:
        return None

    label = " ".join(tag.get_text(" ").split())
    if not label:
        label = tag.get("arialabel") or tag.get("aria-label") or tag.get("title") or ""
    return CandidateLink(url=absolute, label=label[:250], source_url=source_url)


def pdf_text(content: bytes) -> str:
    reader = PdfReader(BytesIO(content))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n\n".join(part for part in parts if part.strip())


def docx_text(content: bytes) -> str:
    with ZipFile(BytesIO(content)) as archive:
        document = archive.read("word/document.xml")

    root = ET.fromstring(document)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        parts = [
            node.text or ""
            for node in paragraph.findall(".//w:t", namespace)
        ]
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


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
