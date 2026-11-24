#!/usr/bin/env python3
"""
Runs a 3-turn conversation with GPT-5 on OpenRouter and saves it as JSON.

Usage:
    export OPENROUTER_API_KEY="your-key-here"     (macOS/Linux)
    setx OPENROUTER_API_KEY "your-key-here"       (Windows, then reopen terminal)
    python run_conversation.py

Pauses after each turn so you can read the reply (and compile/run the code)
before moving on. Saves conversation_run2.json after every turn.
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-5"
OUTPUT_FILE = "conversation_run2.json"

PROMPTS = [
    # Turn 1
    "I'm building a small scheduler library in Java 17 with no external dependencies. "
    "Can you write a class `CronExpression` that parses a standard 5-field cron expression "
    "(minute, hour, day-of-month, month, day-of-week) supporting `*`, lists (`1,15`), "
    "ranges (`1-5`) and steps (`*/10`, `10-30/5`)? It should have a method "
    "`ZonedDateTime next(ZonedDateTime after)` returning the next matching time strictly "
    "after the given one. Treat everything as UTC for now. Throw clear exceptions on "
    "invalid input and include a few JUnit 5 tests. Make sure everything compiles cleanly "
    "with javac 17 and the tests pass.",
    # Turn 2
    "Nice. Now make it respect the time zone of the `ZonedDateTime` passed in "
    "(e.g. `Europe/Lisbon`). Also add the classic cron rule: if both day-of-month and "
    "day-of-week are restricted (not `*`), a date matches if *either* matches. Support "
    "names `JAN-DEC` and `MON-SUN`, and accept `7` as Sunday. Add tests for these.",
    # Turn 3 (the failure target)
    "Now I need it to handle DST the way Vixie cron/cronie does:\n"
    "1. Fixed-time jobs (minute and hour are plain values/lists, not `*` or steps) whose "
    "local time falls inside a spring-forward gap run **once, at the transition instant "
    "itself** (the first valid instant after the gap).\n"
    "2. Fixed-time jobs in a fall-back overlap run **only once**, on the first occurrence.\n"
    "3. Wildcard/step jobs (`*/15 * * * *`, `0 * * * *`) are elapsed-time based: they fire "
    "at every matching slot that actually exists, in **both** passes of a fall-back overlap. "
    "Slots inside a spring-forward gap simply don't occur, and these jobs never fire extra "
    "or catch up afterwards.\n\n"
    "Add a method `List<ZonedDateTime> nextN(ZonedDateTime after, int n)` (strictly after "
    "`after`) and JUnit tests with exact expected instants for `Europe/Lisbon` (2026-03-29 "
    "and 2026-10-25) and `America/New_York` (2026-03-08 and 2026-11-01). Cover "
    "`30 1 * * *`, `30 2 * * *`, `0 * * * *` and `*/30 * * * *`.",
]


def _stream_once(api_key, messages):
    """One streaming request. Prints text as it arrives and returns the full reply."""
    body = json.dumps({"model": MODEL, "messages": messages, "stream": True}).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    parts = []
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            # Blank lines and ": OPENROUTER PROCESSING" comments are keep-alives
            if not line or line.startswith(":") or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if "error" in chunk:
                raise RuntimeError(f"API error: {chunk['error']}")
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {}).get("content")
            if delta:
                parts.append(delta)
                print(delta, end="", flush=True)
    return "".join(parts)


def call_model(api_key, messages, attempts=3):
    for attempt in range(1, attempts + 1):
        try:
            reply = _stream_once(api_key, messages)
            if reply.strip():
                return reply
            print("\n[empty reply, retrying...]")
        except urllib.error.HTTPError as e:
            print(f"\nHTTP {e.code}: {e.read().decode('utf-8', 'replace')}")
            if e.code in (401, 402, 403, 404):  # key, credits or model problems: retrying won't help
                sys.exit(1)
        except (OSError, RuntimeError) as e:
            print(f"\n[connection problem: {e}] retrying ({attempt}/{attempts})...")
        time.sleep(5 * attempt)
    print("\nGave up after several attempts.")
    sys.exit(1)


def save(messages):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "messages": messages}, f, indent=2, ensure_ascii=False)


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Set the OPENROUTER_API_KEY environment variable first.")
        sys.exit(1)

    messages = []
    for i, prompt in enumerate(PROMPTS, start=1):
        print(f"\n=== TURN {i} ===")
        print(prompt, "\n")
        edit = input("Press Enter to send as-is, or type 'e' to paste a modified prompt: ")
        if edit.strip().lower() == "e":
            print("Paste your prompt, then finish with a line containing only END:")
            lines = []
            while True:
                line = input()
                if line == "END":
                    break
                lines.append(line)
            prompt = "\n".join(lines)

        messages.append({"role": "user", "content": prompt})
        print(f"\n--- GPT-5 reply (turn {i}) --- (it may think for a minute before text appears)\n")
        reply = call_model(api_key, messages)
        messages.append({"role": "assistant", "content": reply})
        save(messages)
        print()
        if i < len(PROMPTS):
            input("\nReview the reply (compile/test it if you like), then press Enter for the next turn...")

    print(f"\nDone. Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()