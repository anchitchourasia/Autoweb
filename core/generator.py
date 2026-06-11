def generate_clone_html(data: dict, shot_path: str = "") -> dict:
    from core.agent_chain import run_agent_chain
    # Pass screenshot path so supervisor can do visual analysis
    data["_screenshot_path"] = shot_path
    return run_agent_chain(data)
