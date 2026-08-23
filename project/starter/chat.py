#!/usr/bin/env python3
"""Chat with your support chatbot from the terminal.

    python chat.py

Every run starts ONE conversation (one `runtimeSessionId`). The harness is
stateful: as long as you reuse the same session id, it remembers the whole
conversation — that is what lets it collect bug details over several turns.
Start the script again to get a fresh conversation.

The script attaches your AgentCore Gateway to each invoke, so the model can
call the create_bug_report tool. When it does, you'll see a line like:

    [tool call] bugreports___create_bug_report

Type your message and press Enter. Type 'quit' (or Ctrl-C) to exit.
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.eventstream import EventStream


class ThinkingFilter:
    """Suppress <thinking>...</thinking> blocks from the streamed output.

    Nova emits a reasoning preamble as its own assistant message before a tool
    call. The system prompt tells it not to, and it complies in the final
    reply, but the pre-tool-call message still carries the block — and this
    client prints every delta as it arrives, so the customer would see it.
    Filtering here keeps the display clean without hiding anything the model
    actually says to the customer.
    """

    OPEN, CLOSE = "<thinking>", "</thinking>"

    def __init__(self):
        self.pending = ""
        self.inside = False

    def feed(self, text):
        """Return the portion of `text` that is safe to display."""
        self.pending += text
        out = []
        while self.pending:
            if self.inside:
                end = self.pending.find(self.CLOSE)
                if end == -1:
                    # Keep only enough to recognise a split closing tag.
                    self.pending = self.pending[-len(self.CLOSE):]
                    break
                self.pending = self.pending[end + len(self.CLOSE):]
                self.inside = False
                continue
            start = self.pending.find(self.OPEN)
            if start != -1:
                out.append(self.pending[:start])
                self.pending = self.pending[start + len(self.OPEN):]
                self.inside = True
                continue
            # No tag in sight: emit everything except a possible split tag.
            hold = len(self.OPEN) - 1
            out.append(self.pending[:-hold] if len(self.pending) > hold else "")
            self.pending = self.pending[-hold:] if len(self.pending) > hold else self.pending
            break
        return "".join(out)

    def flush(self):
        """Emit whatever is left once the stream ends."""
        rest = "" if self.inside else self.pending
        self.pending = ""
        return rest


def screen_input(guard, config, text):
    """Run the customer's message past the Bedrock Guardrail.

    Returns None when the message is allowed through, or the canned reply to
    show when the guardrail blocks it. Screening happens BEFORE invoke_harness,
    so a blocked message never reaches the model at all — unlike the system
    prompt's own refusal rules, which the model has to choose to follow.

    Returns None (allow) when no guardrail is configured, so the script still
    works if you skipped setup_guardrail.py.
    """
    if guard is None:
        return None
    result = guard.apply_guardrail(
        guardrailIdentifier=config["guardrail_id"],
        guardrailVersion=str(config["guardrail_version"]),
        source="INPUT",
        content=[{"text": {"text": text}}],
    )
    if result.get("action") != "GUARDRAIL_INTERVENED":
        return None
    # Report which policy fired — useful when tuning the filters.
    reasons = []
    for assessment in result.get("assessments", []):
        for f in assessment.get("contentPolicy", {}).get("filters", []):
            reasons.append(f.get("type", "?"))
    if reasons:
        print(f"[guardrail blocked: {', '.join(sorted(set(reasons)))}]", flush=True)
    outputs = result.get("outputs") or []
    return outputs[0].get("text") if outputs else "Sorry, I can't help with that."


def event_stream(response):
    """Locate the streaming part of the invoke_harness response."""
    for value in response.values():
        if isinstance(value, EventStream):
            return value
    raise RuntimeError(f"No event stream in response: {list(response)}")


def invoke(rt, config, session_id, user_text, verbose=False):
    """Send one user message; print the reply as it streams in.

    Returns the assistant's final text. Tool calls and tool results are
    handled server-side by the harness — we only watch them go by.
    """
    response = rt.invoke_harness(
        harnessArn=config["harness_arn"],
        runtimeSessionId=session_id,
        # Pin the model on every invoke as well (belt and suspenders —
        # create_harness.py already pinned it on the harness).
        model={"bedrockModelConfig": {"modelId": config.get("model_id", "us.amazon.nova-pro-v1:0")}},
        # Attach the gateway so the model can use create_bug_report.
        tools=[{
            "type": "agentcore_gateway",
            "name": "support_gateway",
            "config": {"agentCoreGateway": {"gatewayArn": config["gateway_arn"]}},
        }],
        messages=[{"role": "user", "content": [{"text": user_text}]}],
    )

    texts = []      # completed assistant messages
    buffer = []     # text of the message currently streaming
    thinking = ThinkingFilter()
    for event in event_stream(response):
        if verbose:
            print(f"\n[event] {json.dumps(event, default=str)}", file=sys.stderr)
        if "contentBlockStart" in event:
            tool_use = event["contentBlockStart"].get("start", {}).get("toolUse")
            if tool_use:
                print(f"\n[tool call] {tool_use.get('name', '?')}", flush=True)
        elif "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta", {})
            if "text" in delta:
                visible = thinking.feed(delta["text"])
                if visible:
                    print(visible, end="", flush=True)
                    buffer.append(visible)
        elif "messageStop" in event:
            leftover = thinking.flush()
            if leftover:
                print(leftover, end="", flush=True)
                buffer.append(leftover)
            if buffer:
                texts.append("".join(buffer))
                buffer = []
    leftover = thinking.flush()
    if leftover:
        print(leftover, end="", flush=True)
        buffer.append(leftover)
    if buffer:
        texts.append("".join(buffer))
    print()
    return texts[-1] if texts else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="agentcore_config.json",
                        help="Config file written by the setup scripts.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every raw stream event (for debugging).")
    parser.add_argument("--no-guardrail", action="store_true",
                        help="Skip guardrail screening (to compare behaviour "
                             "with and without it).")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if "harness_arn" not in config:
        sys.exit("No harness in config yet — run create_harness.py first.")

    # Session ids must be at least 33 characters — a UUID plus a suffix.
    session_id = f"{uuid.uuid4()}-support-chat"

    rt = boto3.client(
        "bedrock-agentcore",
        region_name=config["region"],
        # Tool-using turns can take a while; don't let boto3 time out.
        config=Config(read_timeout=300, retries={"max_attempts": 1}),
    )

    # The guardrail is optional: without it the assistant still refuses unsafe
    # requests via the system prompt, it's just soft defence.
    guard = None
    if config.get("guardrail_id") and not args.no_guardrail:
        guard = boto3.client("bedrock-runtime", region_name=config["region"])
        print(f"Guardrail {config['guardrail_id']} "
              f"v{config['guardrail_version']} screening every message.")
    elif not config.get("guardrail_id"):
        print("No guardrail configured (run setup_guardrail.py to add one).")

    print(f"Connected to harness {config.get('harness_name', '?')} "
          f"(session {session_id}).")
    print("Type a message, or 'quit' to exit.\n")

    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not sys.stdin.isatty():
            # Piped input isn't echoed by the terminal; print it so a captured
            # transcript shows both sides of the conversation.
            print(user_text)
        if not user_text:
            continue
        if user_text.lower() in ("quit", "exit"):
            break
        blocked = screen_input(guard, config, user_text)
        if blocked:
            # Never reached the harness; nothing was added to the session.
            print(f"bot> {blocked}\n")
            continue
        print("bot> ", end="", flush=True)
        invoke(rt, config, session_id, user_text, verbose=args.verbose)
        print()


if __name__ == "__main__":
    main()
