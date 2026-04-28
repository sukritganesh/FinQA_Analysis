#!/usr/bin/env python3
"""Serve a local vLLM model for one prompt, then shut it down.

This helper starts `vllm serve` inside a target virtual environment,
waits for the server to become ready, sends a single chat prompt to the
OpenAI-compatible `/v1/chat/completions` endpoint, prints the model
response, and then stops the server process.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_ENV_DIR = ".finqa-vllm-env"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Serve a local vLLM model, run one prompt, then stop the server.",
    )
    parser.add_argument(
        "prompt",
        help="Prompt text to send to the model.",
    )
    parser.add_argument(
        "--env-dir",
        default=DEFAULT_ENV_DIR,
        help=f"Virtual environment directory containing vLLM. Default: {DEFAULT_ENV_DIR}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name to serve. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="vLLM host. Default: 127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM port. Default: 8000",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum output tokens for the prompt. Default: 512",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=180,
        help="Seconds to wait for the vLLM server to become ready. Default: 180",
    )
    parser.add_argument(
        "--log-file",
        default="scripts/vllm/vllm_server.log",
        help="Path to the vLLM server log file. Default: scripts/vllm/vllm_server.log",
    )
    return parser.parse_args()


def build_activate_command(env_dir: Path, model: str, host: str, port: int) -> str:
    """Build the shell command that activates the environment and starts vLLM."""
    activate_candidates = [
        env_dir / "bin" / "activate",
        env_dir / "Scripts" / "activate",
    ]
    activate_path = next((path for path in activate_candidates if path.exists()), None)
    if activate_path is None:
        raise FileNotFoundError(
            "Could not find an activation script under either "
            f"{env_dir / 'bin' / 'activate'} or {env_dir / 'Scripts' / 'activate'}."
        )
    return f"source {shell_quote(str(activate_path))} && vllm serve {shell_quote(model)} --host {host} --port {port}"


def shell_quote(value: str) -> str:
    """Quote a value for safe inclusion in a simple bash command."""
    return "'" + value.replace("'", "'\"'\"'") + "'"


def wait_for_server(
    base_url: str,
    timeout_seconds: int,
    process: subprocess.Popen[str],
    log_file: Path,
) -> None:
    """Wait until the OpenAI-compatible models endpoint responds successfully."""
    deadline = time.time() + timeout_seconds
    models_url = f"{base_url}/v1/models"
    last_error: Exception | None = None

    while time.time() < deadline:
        if process.poll() is not None:
            log_tail = read_log_tail(log_file)
            raise RuntimeError(
                "vLLM server exited before becoming ready.\n"
                f"Log tail:\n{log_tail}"
            )
        try:
            request = urllib.request.Request(models_url, method="GET")
            with urllib.request.urlopen(request, timeout=5) as response:
                if 200 <= response.status < 300:
                    return
        except Exception as exc:  # noqa: BLE001 - surfacing startup failures is enough here.
            last_error = exc
            time.sleep(2)

    raise TimeoutError(
        f"Timed out waiting for vLLM server at {models_url}. Last error: {last_error}"
    )


def send_prompt(base_url: str, model: str, prompt: str, max_tokens: int) -> dict:
    """Send a single chat completion request."""
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM request failed with HTTP {exc.code}: {details}") from exc


def extract_text(response_json: dict) -> str:
    """Extract assistant text from a chat completions response."""
    choices = response_json.get("choices") or []
    if not choices:
        raise ValueError(f"Response did not contain choices: {response_json}")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError(f"Response did not contain text content: {response_json}")
    return content.strip()


def terminate_process(process: subprocess.Popen[str]) -> None:
    """Stop the vLLM server process group cleanly."""
    if process.poll() is not None:
        return

    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        process.terminate()
    try:
        process.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        pass

    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    else:
        process.kill()
    process.wait(timeout=10)


def read_log_tail(log_file: Path, max_lines: int = 40) -> str:
    """Read the last few lines of the log file for debugging."""
    if not log_file.exists():
        return "<log file does not exist>"

    lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return "<log file is empty>"
    return "\n".join(lines[-max_lines:])


def main() -> int:
    """Run the one-shot vLLM workflow."""
    args = parse_args()
    env_dir = Path(args.env_dir).expanduser().resolve()
    base_url = f"http://{args.host}:{args.port}"
    log_file = Path(args.log_file).expanduser().resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    command = build_activate_command(
        env_dir=env_dir,
        model=args.model,
        host=args.host,
        port=args.port,
    )

    with log_file.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            ["bash", "-lc", command],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=(os.name != "nt"),
        )

        try:
            wait_for_server(
                base_url=base_url,
                timeout_seconds=args.startup_timeout,
                process=process,
                log_file=log_file,
            )
            response_json = send_prompt(
                base_url=base_url,
                model=args.model,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
            )
            print(extract_text(response_json))
            return 0
        finally:
            terminate_process(process)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
