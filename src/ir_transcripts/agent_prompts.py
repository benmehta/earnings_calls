from __future__ import annotations

from dataclasses import dataclass
from importlib import resources


@dataclass(frozen=True)
class AgentPrompt:
    name: str
    description: str
    system_prompt: str
    human_prompt: str | None = None


def load_agent_prompt(name: str) -> AgentPrompt:
    """Load a small YAML prompt config bundled with the package."""
    path = resources.files("ir_transcripts").joinpath("agents", f"{name}.yaml")
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing agent prompt config: {name}") from exc
    data = parse_prompt_yaml(text)
    try:
        return AgentPrompt(
            name=str(data["name"]),
            description=str(data.get("description", "")),
            system_prompt=str(data["system_prompt"]).strip(),
            human_prompt=str(data["human_prompt"]).strip() if data.get("human_prompt") else None,
        )
    except KeyError as exc:
        raise RuntimeError(f"Agent prompt config {name} is missing required key {exc}") from exc


def parse_prompt_yaml(text: str) -> dict[str, str]:
    """Parse the tiny YAML subset used by prompt configs.

    Supported forms:
      key: value
      key: |
        multiline value

    This avoids adding a PyYAML dependency just for static prompt text.
    """
    result: dict[str, str] = {}
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        index += 1
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in raw:
            raise ValueError(f"Invalid prompt YAML line: {raw!r}")
        key, value = raw.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid empty prompt YAML key: {raw!r}")
        if value == "|":
            block: list[str] = []
            while index < len(lines):
                block_line = lines[index]
                if block_line and not block_line.startswith((" ", "\t")):
                    break
                block.append(block_line[2:] if block_line.startswith("  ") else block_line.lstrip())
                index += 1
            result[key] = "\n".join(block).strip()
        else:
            result[key] = value.strip("'\"")
    return result
