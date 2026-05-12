class AgentExecutionError(RuntimeError):
    """Raised when an agent cannot produce a valid model-backed decision."""

    def __init__(self, agent: str, code: str, detail: str | None = None):
        self.agent = agent
        self.code = code
        self.detail = detail
        message = f"{agent} agent failed closed: {code}"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)
