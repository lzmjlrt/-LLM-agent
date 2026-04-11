try:
    from agent.app.chat_app import run_streamlit_app
except ModuleNotFoundError:
    import os
    import sys

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from agent.app.chat_app import run_streamlit_app


if __name__ == "__main__":
    run_streamlit_app()
