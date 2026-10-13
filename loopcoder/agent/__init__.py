"""agent 主循环与会话编排层。"""

__all__ = ["AgentLoop", "AgentSession"]


def __getattr__(name: str):
    if name == "AgentLoop":
        from loopcoder.agent.loop import AgentLoop

        return AgentLoop
    if name == "AgentSession":
        from loopcoder.agent.session import AgentSession

        return AgentSession
    raise AttributeError(name)
