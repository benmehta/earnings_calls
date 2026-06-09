from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


TRACKING_PREFIXES = ("utm_",)
TRACKING_PARAMS = {"fbclid", "gclid", "mc_cid", "mc_eid"}
CURATED_DOCUMENT_REDIRECTS = {
    "https://aka.ms/transcriptfy26q3": "https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/TranscriptQandAFY26Q3",
}


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")

    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key not in TRACKING_PARAMS and not key.startswith(TRACKING_PREFIXES)
    ]
    query = urlencode(sorted(query_pairs), doseq=True)
    return urlunparse((scheme, netloc, path, "", query, ""))


def resolve_document_url(url: str) -> str:
    redirected = CURATED_DOCUMENT_REDIRECTS.get(url)
    if redirected:
        return redirected

    parsed = urlparse(url)
    if parsed.netloc.lower() == "view.officeapps.live.com" and parsed.path.lower() == "/op/view.aspx":
        params = dict(parse_qsl(parsed.query, keep_blank_values=True))
        source = params.get("src")
        if source and urlparse(source).scheme in {"http", "https"}:
            return source

    return url


def same_origin(url: str, origin_url: str) -> bool:
    return urlparse(url).netloc.lower() == urlparse(origin_url).netloc.lower()


def host(url: str) -> str:
    return urlparse(url).netloc.lower()
