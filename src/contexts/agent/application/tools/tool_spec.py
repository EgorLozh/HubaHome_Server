from dataclasses import dataclass
from typing import Awaitable, Callable, Literal


ToolKind = Literal["read", "write", "external_side_effect"]


@dataclass
class ToolCall:
    name: str
    input: dict


@dataclass
class ToolResult:
    tool_name: str
    ok: bool
    message: str
    data: dict


@dataclass
class ToolSpec:
    name: str
    description: str
    tool_kind: ToolKind
    requires_confirmation: bool
    execute: Callable[[dict], Awaitable[ToolResult]]
