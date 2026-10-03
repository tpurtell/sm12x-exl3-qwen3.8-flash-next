#!/usr/bin/env python3
"""Synthetic SSE benchmark for decode versus SSD-backed PLE prefill interference.

Run baseline and candidate with equivalent recorded cache preparation with the same
--seed and --run-id. No prompt/response text or API keys are written to results.
Chunk arrivals are NOT token timestamps; server usage counts are kept separately.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import random
import socket
import threading
import time
from urllib.parse import urlparse


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def synthetic_prompt(words, seed, tag):
    rng = random.Random(f"{seed}:{tag}")
    # The nonce is at the beginning to prevent reuse across cases in this run.
    rows = " ".join(f"record{rng.randrange(10**9):09d}" for _ in range(words))
    return (f"Synthetic benchmark {tag}.\n{rows}\n"
            "Discuss patterns in the synthetic records at length, continuing until the output limit.")


def stream_request(args, prompt, label, first_chunk=None, barrier=None):
    parsed = urlparse(args.base_url)
    cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    conn = cls(parsed.hostname, parsed.port, timeout=args.timeout)
    payload = {
        "model": args.model, "messages": [{"role": "user", "content": prompt}],
        "stream": True, "stream_options": {"include_usage": True},
        "max_tokens": args.output_tokens, "min_tokens": args.output_tokens,
        "ignore_eos": True, "temperature": 0, "seed": args.seed,
        "chat_template_kwargs": {"enable_thinking": args.thinking},
    }
    headers = {"Content-Type": "application/json"}
    if os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]
    result = {"label": label, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
              "prompt_characters": len(prompt), "chunks": [], "usage": None,
              "finish_reason": None, "error": None, "done_received": False}
    if barrier:
        barrier.wait(timeout=args.timeout)
    started = time.perf_counter()
    result["started_unix_s"] = time.time()
    expired = threading.Event()
    active_socket = [None]

    def abort():
        expired.set()
        sock = active_socket[0] or conn.sock
        if sock:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        conn.close()

    timer = threading.Timer(args.timeout, abort)
    timer.daemon = True
    timer.start()
    try:
        conn.request("POST", parsed.path.rstrip("/") + "/chat/completions",
                     body=json.dumps(payload), headers=headers)
        active_socket[0] = conn.sock
        response = conn.getresponse()
        if response.status != 200:
            # Do not retain an error body that could echo credentials or prompts.
            raise RuntimeError(f"HTTP {response.status}")
        pending = []
        while True:
            raw = response.readline()
            if not raw:
                break
            if expired.is_set():
                raise TimeoutError("request deadline exceeded")
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("data:"):
                pending.append(line[5:].lstrip())
            if line or not pending:
                continue
            data = "\n".join(pending)
            pending.clear()
            if data == "[DONE]":
                result["done_received"] = True
                break
            event = json.loads(data)
            observed = time.perf_counter() - started
            if "error" in event:
                raise RuntimeError("server returned an SSE error event")
            if isinstance(event.get("usage"), dict):
                result["usage"] = event["usage"]
            for choice in event.get("choices", []):
                if choice.get("finish_reason") is not None:
                    result["finish_reason"] = choice["finish_reason"]
                delta = choice.get("delta") or {}
                fields = [key for key in ("content", "reasoning_content", "reasoning")
                          if isinstance(delta.get(key), str) and delta[key]]
                if fields:
                    result["chunks"].append({"elapsed_s": observed,
                                             "fields": fields,
                                             "characters": sum(len(delta[k]) for k in fields)})
                    if first_chunk:
                        first_chunk.set()
        if not result["done_received"]:
            raise RuntimeError("stream ended without [DONE]")
    except Exception as exc:
        # Exception details from the server are intentionally not persisted.
        if expired.is_set() or isinstance(exc, TimeoutError):
            result["error"] = "request deadline exceeded"
        elif isinstance(exc, RuntimeError):
            result["error"] = type(exc).__name__ + ": " + str(exc)
        else:
            result["error"] = type(exc).__name__
    finally:
        timer.cancel()
        conn.close()
    result["wall_s"] = time.perf_counter() - started
    arrivals = [c["elapsed_s"] for c in result["chunks"]]
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    result.update({"ttft_s": arrivals[0] if arrivals else None,
                   "max_inter_chunk_gap_s": max(gaps) if gaps else None,
                   "p95_inter_chunk_gap_s": percentile(gaps, .95),
                   "chunk_count": len(arrivals), "status": "error" if result["error"] else "completed"})
    usage_tokens = (result["usage"] or {}).get("completion_tokens")
    result["usage_completion_tokens_per_wall_s"] = usage_tokens / result["wall_s"] if isinstance(usage_tokens, (float, int)) else None
    return result


def run_case(args, case, repeat):
    tag = f"{args.run_id}-{case}-{repeat}"
    prompts = [synthetic_prompt(args.short_words, args.seed, tag + f"-decode-{i}") for i in range(2)]
    started = time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        if case in ("c1", "c2"):
            n = 1 if case == "c1" else 2
            barrier = threading.Barrier(n)
            tasks = [pool.submit(stream_request, args, prompts[i], f"decode-{i}", None, barrier)
                     for i in range(n)]
            results = [f.result() for f in tasks]
            valid, reason = all(r["status"] == "completed" for r in results), None
        else:
            ready = threading.Event()
            decode = pool.submit(stream_request, args, prompts[0], "ongoing-decode", ready)
            if not ready.wait(args.timeout):
                return {"case": case, "repeat": repeat, "valid": False,
                        "invalid_reason": "decode produced no chunk before deadline", "requests": [decode.result()]}
            time.sleep(args.mixed_delay)
            if decode.done():
                return {"case": case, "repeat": repeat, "valid": False,
                        "invalid_reason": "decode finished before long prompt; increase output tokens or reduce mixed delay",
                        "requests": [decode.result()]}
            prompt = synthetic_prompt(args.long_words, args.seed, tag + "-long")
            long_result = stream_request(args, prompt, "long-prefill")
            decode_result = decode.result()
            results = [decode_result, long_result]
            long_start = long_result["started_unix_s"]
            decode_end = decode_result["started_unix_s"] + decode_result["wall_s"]
            valid = long_start < decode_end and all(r["status"] == "completed" for r in results)
            reason = None if valid else "requests did not overlap or at least one failed"
    return {"case": case, "repeat": repeat, "started_unix_s": started,
            "wall_s": time.time() - started, "valid": valid, "invalid_reason": reason,
            "requests": results}


def self_test():
    """Exercise real local HTTP/SSE parsing, overlap, usage, EOF and deadline handling."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *unused):
            pass
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                if self.path.startswith("/timeout"):
                    time.sleep(.3)
                    return
                for i in range(12):
                    data = {"choices": [{"delta": {"reasoning_content": "synthetic"}}]}
                    self.wfile.write(("data: " + json.dumps(data) + "\n\n").encode())
                    self.wfile.flush()
                    time.sleep(.01)
                if self.path.startswith("/truncated"):
                    return
                self.wfile.write(b'data: {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 24}}\n\ndata: [DONE]\n\n')
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    args = argparse.Namespace(base_url=f"http://127.0.0.1:{server.server_port}/v1", timeout=3,
                              model="fake", output_tokens=24, seed=7, thinking=True,
                              run_id="selftest", short_words=2, long_words=8, mixed_delay=.01)
    try:
        for case in ("c1", "c2", "mixed"):
            result = run_case(args, case, 0)
            assert result["valid"], result
            for req in result["requests"]:
                assert req["chunk_count"] == 12 and req["usage"]["completion_tokens"] == 24
                assert req["p95_inter_chunk_gap_s"] > 0 and req["ttft_s"] > 0
        args.base_url = f"http://127.0.0.1:{server.server_port}/truncated"
        assert stream_request(args, "synthetic", "truncated")["status"] == "error"
        args.base_url = f"http://127.0.0.1:{server.server_port}/timeout"
        args.timeout = .05
        assert stream_request(args, "synthetic", "timeout")["error"] == "request deadline exceeded"
        args.base_url = f"http://127.0.0.1:{server.server_port}/v1"
        trickle = stream_request(args, "synthetic", "trickle-timeout")
        assert trickle["error"] == "request deadline exceeded" and trickle["wall_s"] < .25
        assert synthetic_prompt(8, 7, "a") == synthetic_prompt(8, 7, "a")
        assert synthetic_prompt(8, 7, "a") != synthetic_prompt(8, 7, "b")
    finally:
        server.shutdown()
        server.server_close()
    print("PASS: local SSE C1/C2/mixed, usage vs chunks, EOF, deadline, deterministic inputs")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="qwen3.8-flash-next-fast")
    p.add_argument("--label", default="unlabelled", help="baseline or candidate plus runtime commit")
    p.add_argument("--run-id", default="ple-comparison-1", help="use identical value across matched baseline/candidate runs")
    p.add_argument("--seed", type=int, default=20261002)
    p.add_argument("--short-words", type=int, default=16)
    p.add_argument("--long-words", type=int, default=4096, help="synthetic records, NOT token count; inspect usage")
    p.add_argument("--output-tokens", type=int, default=512)
    p.add_argument("--timeout", type=float, default=600, help="total per-request deadline in seconds")
    p.add_argument("--mixed-delay", type=float, default=.5, help="seconds after ongoing decode's first chunk")
    p.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cases", nargs="+", choices=("c1", "c2", "mixed"), default=["c1", "c2", "mixed"])
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--output", type=Path, default=Path("ple-interference-results.json"))
    p.add_argument("--self-test", action="store_true", help="only contact an ephemeral local fake HTTP server")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if any(v <= 0 for v in (args.short_words, args.long_words, args.output_tokens, args.timeout, args.repeats)) or args.mixed_delay < 0:
        p.error("sizes, timeout and repeats must be positive; mixed delay must be nonnegative")
    parsed = urlparse(args.base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        p.error("base URL must be HTTP(S), with no embedded credentials, query, or fragment")
    config = vars(args).copy()
    config["output"] = str(args.output)
    report = {"schema_version": 1, "config": config,
              "notes": ["Arrivals describe SSE text chunks, not individual tokens.",
                        "Usage tokens are server-reported; prompts and generated text are omitted.",
                        "Short-prompt C1/C2 approximate decode workloads; TTFT includes their prefill.",
                        "Cache state is not controlled by this script. Restart equivalently for matched runs.",
                        "No automatic throughput acceptance claim: inspect all request errors and overlap."],
              "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for repeat in range(args.repeats):
        for case in args.cases:
            print(f"Running {case} repeat {repeat + 1}/{args.repeats}", flush=True)
            result = run_case(args, case, repeat)
            report["cases"].append(result)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"case": case, "valid": result["valid"],
                              "requests": [{k: r[k] for k in ("label", "status", "ttft_s", "max_inter_chunk_gap_s", "p95_inter_chunk_gap_s", "usage", "wall_s")} for r in result["requests"]]}), flush=True)
    if not all(case["valid"] for case in report["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
