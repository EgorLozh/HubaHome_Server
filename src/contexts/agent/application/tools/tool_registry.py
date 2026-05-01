from src.contexts.agent.application.tools.tool_spec import ToolResult, ToolSpec


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec]) -> None:
        self._specs = {spec.name: spec for spec in specs}

    @property
    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    async def call(self, name: str, payload: dict) -> ToolResult:
        spec = self.get(name)
        if spec is None:
            return ToolResult(
                tool_name=name,
                ok=False,
                message=f"Unknown tool: {name}",
                data={},
            )
        return await spec.execute(payload)
