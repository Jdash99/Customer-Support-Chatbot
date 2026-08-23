#!/usr/bin/env python3
"""Create the Bedrock Guardrail that screens customer messages.

    python setup_guardrail.py

Why a guardrail on top of the system prompt: the prompt already tells the
assistant to refuse harmful requests and to ignore instructions that try to
change its role, and in testing it did. But that is soft defence — the model
decides, every time, whether to obey. A guardrail is enforced outside the
model: the message is screened BEFORE the harness ever sees it, so a blocked
message cannot influence the assistant at all.

The harness API has no guardrail field, so the guardrail is applied
client-side with `bedrock-runtime.apply_guardrail` (see chat.py). That is
also what puts it genuinely "before any model processes the message".

The guardrail id and version are saved into agentcore_config.json.
"""

import argparse
import json
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# Everything the shop's support chat should never have to process. PROMPT_ATTACK
# is input-only by design — it catches "ignore your instructions"-style messages.
CONTENT_FILTERS = [
    {"type": "PROMPT_ATTACK", "inputStrength": "HIGH", "outputStrength": "NONE"},
    {"type": "HATE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
    {"type": "INSULTS", "inputStrength": "MEDIUM", "outputStrength": "MEDIUM"},
    {"type": "SEXUAL", "inputStrength": "HIGH", "outputStrength": "HIGH"},
    {"type": "VIOLENCE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
    {"type": "MISCONDUCT", "inputStrength": "HIGH", "outputStrength": "HIGH"},
]

# A blocked customer still deserves somewhere to go, so the canned message is
# the same hand-off the assistant would give — phone number included.
BLOCKED_INPUT = (
    "I'm sorry, that's not something I can handle from this chat. "
    "Please call our support line at 1-800-555-0199, Monday to Friday."
)
BLOCKED_OUTPUT = BLOCKED_INPUT


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="support-chatbot-guardrail")
    p.add_argument("--config", default="agentcore_config.json")
    args = p.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"{args.config} not found — run setup_gateway.py first.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    region = config["region"]

    br = boto3.client("bedrock", region_name=region)

    # Reuse the guardrail if it already exists, so re-running is safe.
    existing = None
    for g in br.list_guardrails().get("guardrails", []):
        if g.get("name") == args.name:
            existing = g
            break

    if existing:
        guardrail_id = existing["id"]
        print(f"Guardrail '{args.name}' already exists ({guardrail_id}) — updating...")
        br.update_guardrail(
            guardrailIdentifier=guardrail_id,
            name=args.name,
            description="Screens customer messages for harmful content and prompt injection.",
            contentPolicyConfig={"filtersConfig": CONTENT_FILTERS},
            blockedInputMessaging=BLOCKED_INPUT,
            blockedOutputsMessaging=BLOCKED_OUTPUT,
        )
    else:
        print(f"Creating guardrail '{args.name}'...")
        try:
            created = br.create_guardrail(
                name=args.name,
                description="Screens customer messages for harmful content and prompt injection.",
                contentPolicyConfig={"filtersConfig": CONTENT_FILTERS},
                blockedInputMessaging=BLOCKED_INPUT,
                blockedOutputsMessaging=BLOCKED_OUTPUT,
            )
        except ClientError as exc:
            sys.exit(f"create_guardrail failed: {exc}")
        guardrail_id = created["guardrailId"]

    # A DRAFT guardrail can be applied directly, but a numbered version is
    # stable — publish one so the client always screens against the same rules.
    version = br.create_guardrail_version(
        guardrailIdentifier=guardrail_id,
        description="Version used by chat.py.",
    )["version"]

    config["guardrail_id"] = guardrail_id
    config["guardrail_version"] = version
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    print(f"  guardrail id: {guardrail_id}")
    print(f"  version:      {version}")
    print(f"\nSaved to {args.config}. chat.py will screen every message with it.")


if __name__ == "__main__":
    main()
