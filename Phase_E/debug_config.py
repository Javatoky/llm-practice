from dataclasses import dataclass


@dataclass
class AgentResult:
    answer: str
    called_tools: list[str]