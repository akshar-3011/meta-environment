"""Enhanced inference script backed by shared inference runner."""

from core.inference import run_agent


def run_enhanced_agent():
    run_agent(
        reveal_label=True,
        title="ENHANCED WORKPLACE ENVIRONMENT AGENT",
    )


if __name__ == "__main__":
    run_enhanced_agent()
