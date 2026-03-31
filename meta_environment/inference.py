"""Basic inference script backed by shared inference runner."""

from core.inference import run_agent


if __name__ == "__main__":
    run_agent(
        reveal_label=False,
        title="WORKPLACE ENVIRONMENT AGENT",
    )