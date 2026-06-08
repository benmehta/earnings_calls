from __future__ import annotations

import pandas as pd

from .models import Company


SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def load_sp500() -> list[Company]:
    """Load the current S&P 500 table at runtime.

    This avoids baking a stale company list into the repo. It requires network
    access when the scraper is actually run.
    """
    tables = pd.read_html(SP500_WIKIPEDIA_URL)
    constituents = tables[0]

    companies: list[Company] = []
    for row in constituents.to_dict("records"):
        companies.append(
            Company(
                symbol=str(row["Symbol"]).replace(".", "-").strip(),
                name=str(row["Security"]).strip(),
                sector=str(row.get("GICS Sector", "")).strip() or None,
                sub_industry=str(row.get("GICS Sub-Industry", "")).strip() or None,
                cik=str(row.get("CIK", "")).strip().zfill(10) or None,
            )
        )
    return companies

