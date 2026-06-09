from core.agent_chain import run_agent_chain


def generate_clone_html(data: dict) -> dict:
    return run_agent_chain(data)