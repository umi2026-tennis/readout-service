import anthropic

from readout_service.stats import ab_test


def summarize(result):
    """Ask Claude to turn ab_test results into a short plain-English readout."""
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    prompt = (
        "You are a data scientist writing a short A/B test readout for a product team. "
        "In 3-4 plain-English sentences, say what happened, whether the result is "
        "statistically significant, and what the team should do next.\n\n"
        f"Results: {result}"
    )

    response = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=16000,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": prompt}],
    )

    # NEW: show how many tokens this call used and what it cost
    usage = response.usage
    cost = usage.input_tokens / 1_000_000 * 0.10 + usage.output_tokens / 1_000_000 * 0.50
    print(f"input tokens: {usage.input_tokens}, output tokens: {usage.output_tokens}")
    print(f"cost: ${cost:.5f}")

    if response.stop_reason == "refusal":
        return "Claude declined to write this readout."
    return "".join(block.text for block in response.content if block.type == "text")


def main():
    result = ab_test(control=(100, 1000), treatment=(120, 1000))
    print(summarize(result))


if __name__ == "__main__":
    main()