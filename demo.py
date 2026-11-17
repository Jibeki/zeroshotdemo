"""Compare direct and two-stage zero-shot prompting with local Ollama.

This demo uses the juggler and boxing examples from Figures 1 and 2 of
Kojima et al. (2022) and follows their two-stage method. The GPT-2 XL
option uses an 8-bit GGUF conversion of the checkpoint listed in Table 8.
"""

import argparse
import json
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MODEL = "qwen2.5:3b"
GPT2_MODEL = "hf.co/ddh0/GPT-2-XL-GGUF:Q8_0"
OLLAMA_URL = "http://127.0.0.1:11434/api"
QUESTIONS = (
    (
        "A juggler has 16 balls. Half are golf balls, and half of those are blue. "
        "How many blue golf balls does the juggler have?",
        4,
    ),
    (
        "On average Joe throws 25 punches per minute. A fight lasts 5 rounds of 3 minutes. "
        "How many punches did he throw?",
        375,
    ),
)
REASONING_CUE = "Let's think step by step."
ANSWER_FORMAT = "Reply with exactly one number."
DIRECT_CUE = "The answer (arabic numerals) is"
EXTRACTION_CUE = "Therefore, the answer (arabic numerals) is"


def request_ollama(endpoint, model, **generation):
    """Send a request to the local Ollama server and read its JSON response."""
    payload = {"model": model, "stream": False, **generation}
    request = Request(
        f"{OLLAMA_URL}/{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=90) as response:
            return json.load(response)
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        if error.code == 404 and "model" in details.lower():
            raise SystemExit(f"Download the model first with `ollama pull {model}`.")
        raise SystemExit(f"Ollama returned HTTP {error.code}: {details}")


def ask(messages):
    """Send chat messages to Qwen and return the reply."""
    result = request_ollama(
        "chat", MODEL, messages=messages,
        options={"temperature": 0, "num_predict": 200},
    )
    reply = result["message"]["content"].strip()
    if not reply:
        raise SystemExit("Ollama returned an empty reply. Try running the demo again.")
    return reply


def run_qwen_demo(question, expected_answer):
    print(f"Model: {MODEL}\n\n{question}")

    print(f'\n1. Standard zero-shot\nPrompt cue\n"{ANSWER_FORMAT}"', flush=True)
    direct_prompt = {"role": "user", "content": f"{question}\n\n{ANSWER_FORMAT}"}
    print(f"Model answer\n{ask([direct_prompt])}", flush=True)

    print(f'\n2a. Zero-shot CoT (generate steps)\nPrompt cue\n"{REASONING_CUE}"', flush=True)
    reasoning_prompt = {
        "role": "user",
        "content": f"{question}\n\n{REASONING_CUE}",
    }
    steps = ask([reasoning_prompt])
    print(f"Generated steps\n{steps}", flush=True)

    print(
        "\n2b. Zero-shot CoT (extract the final answer)\n"
        "The model gets the original question, the step-by-step cue, "
        "and the steps above.\n"
        f'Final instruction\n"{ANSWER_FORMAT}"',
        flush=True,
    )
    # Supply the steps from 2a as this model's previous reply to this question.
    answer = ask([
        reasoning_prompt,
        {"role": "assistant", "content": steps},
        {"role": "user", "content": ANSWER_FORMAT},
    ])
    print(f"Model answer\n{answer}")
    print(f"\nExpected answer\n{expected_answer}", flush=True)


def complete(prompt, max_tokens):
    """Let GPT-2 XL continue the supplied prompt as plain text."""
    result = request_ollama(
        "generate", GPT2_MODEL, prompt=prompt, raw=True,
        options={
            "temperature": 0,
            "repeat_penalty": 1.0,
            "num_predict": max_tokens,
            "stop": ["Q:"],
        },
    )
    return result["response"]


def show_numeric_answer(output):
    """Display the raw continuation and take its first number as the answer."""
    match = re.search(r"-?\d+(?:\.\d+)?", output.replace(",", ""))
    print(f"Model output\n{output.strip() or '(empty)'}")
    print(f"Extracted answer\n{match.group() if match else '(no number found)'}", flush=True)


def run_gpt2_demo(question, expected_answer):
    print(f"Model: GPT-2 XL (8-bit GGUF)\n\n{question}")
    question_prompt = f"Q: {question}\nA:"

    print(f'\n1. Standard zero-shot\nPrompt cue\n"{DIRECT_CUE}"', flush=True)
    show_numeric_answer(complete(f"{question_prompt} {DIRECT_CUE}", max_tokens=32))

    print(f'\n2a. Zero-shot CoT (generate steps)\nPrompt cue\n"{REASONING_CUE}"', flush=True)
    reasoning_prompt = f"{question_prompt} {REASONING_CUE}"
    steps = complete(reasoning_prompt, max_tokens=128)
    print(f"Generated steps\n{steps.strip() or '(empty)'}", flush=True)

    print(
        "\n2b. Zero-shot CoT (extract the final answer)\n"
        "The model gets the original prompt followed by the steps above.\n"
        f'Answer cue\n"{EXTRACTION_CUE}"',
        flush=True,
    )
    extraction_prompt = f"{reasoning_prompt}{steps} \n{EXTRACTION_CUE}"
    show_numeric_answer(complete(extraction_prompt, max_tokens=32))
    print(f"\nExpected answer\n{expected_answer}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--model", choices=[MODEL, "gpt2-xl"], default=MODEL)
    args = parser.parse_args()
    if args.model == "gpt2-xl":
        run_demo = run_gpt2_demo
    else:
        run_demo = run_qwen_demo

    for number, (question, expected_answer) in enumerate(QUESTIONS, start=1):
        if number > 1:
            try:
                input(f"\nPress Enter to run Question {number}... ")
            except (EOFError, KeyboardInterrupt):
                print()
                return
        print(f"\nQuestion {number} of {len(QUESTIONS)}")
        run_demo(question, expected_answer)


if __name__ == "__main__":
    try:
        main()
    except TimeoutError:
        raise SystemExit("Ollama took too long to respond. Run the demo again after the model loads.")
    except URLError as error:
        raise SystemExit(f"Could not reach Ollama. Open the app and try again. ({error.reason})")
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        raise SystemExit("Ollama returned an unexpected chat response.")
