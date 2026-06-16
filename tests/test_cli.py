from ir_transcripts.cli import build_parser, load_requested_companies
from ir_transcripts.models import Company


def test_load_requested_companies_appends_non_sp500_symbol(monkeypatch) -> None:
    monkeypatch.setattr("ir_transcripts.cli.load_sp500", lambda: [Company(symbol="NVDA", name="NVIDIA")])

    companies = load_requested_companies(["TSM"])

    assert companies == [
        Company(symbol="TSM", name="Taiwan Semiconductor Manufacturing Company")
    ]


def test_load_requested_companies_keeps_requested_sp500_symbol(monkeypatch) -> None:
    monkeypatch.setattr("ir_transcripts.cli.load_sp500", lambda: [Company(symbol="NVDA", name="NVIDIA")])

    companies = load_requested_companies(["NVDA"])

    assert companies == [Company(symbol="NVDA", name="NVIDIA")]


def test_load_requested_companies_keeps_unknown_symbol_name_empty(monkeypatch) -> None:
    monkeypatch.setattr("ir_transcripts.cli.load_sp500", lambda: [])

    companies = load_requested_companies(["AMZN"])

    assert companies == [Company(symbol="AMZN", name=None)]


def test_disable_memory_flag_is_available() -> None:
    args = build_parser().parse_args(["--disable-memory"])

    assert args.disable_memory is True


def test_no_memory_write_flag_is_available() -> None:
    args = build_parser().parse_args(["--no-memory-write"])

    assert args.no_memory_write is True


def test_default_discovery_is_homepage_first_without_research_agent() -> None:
    args = build_parser().parse_args([])

    assert args.discovery_mode == "nav-first"
    assert args.research_agent is False
