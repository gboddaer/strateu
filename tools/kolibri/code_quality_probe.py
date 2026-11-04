#!/usr/bin/env python3
"""Local diagnostic: paired greedy generation on identical forced prompt IDs.

Not a general benchmark. Cases are JSON records with tag, system, user, cap,
and optional C++ prefix/suffix for testing unedited generated functions.
Requires an already running llama.cpp server. By default also loads Strata;
--llama-only measures a separate reference without loading another engine.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from serve.frontend import ChatTemplate
import strata_tokenizer as ST
from reuse_gate import Engine


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as response:
        return json.load(response)


def tokenizer(path):
    path = Path(path)
    vocab = json.loads((path / "vocab.json").read_text())
    tokens = [None] * len(vocab)
    for token, idx in vocab.items():
        tokens[idx] = token
    cfg = json.loads((path / "tokenizer.json").read_text())
    return ST.Tokenizer(tokens, (path / "merges.txt").read_text().split("\n"),
                        json.loads((path / "token_type.json").read_text()),
                        cfg.get("pre") or "qwen35", cfg.get("special_ids") or None)


def source_of(text):
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    blocks = re.findall(r"```(?:cpp|c\+\+)?\s*\n(.*?)```", text, re.S)
    if blocks:
        return "\n".join(blocks)
    # Keep plain code unedited. Strip only the model's terminal EOS marker.
    return text.split("<|im_end|>", 1)[0]


def evaluate(out, tag, text, case):
    if case.get("enable_thinking", False) and "</think>" not in text:
        return {"no_answer": True, "reasoning_closed": False}
    source = source_of(text)
    cpp = out / (tag + ".cpp")
    cpp.write_text(case.get("prefix", "") + source + case.get("suffix", ""))
    result = {"source_lines": len(source.splitlines())}
    if not case.get("compile", True):
        return result
    binary = out / (tag + ".bin")
    # `openpty` lives in libutil on glibc < 2.34, so the PTY oracle must link it there;
    # newer glibc keeps a stub, so the flag is harmless where it is already in libc.
    compile_ = subprocess.run(["g++", "-std=c++17", "-O2", "-o", str(binary), str(cpp), "-lutil"],
                              capture_output=True, text=True, timeout=60)
    (out / (tag + "-compile.log")).write_text(compile_.stdout + compile_.stderr)
    result["compile_exit"] = compile_.returncode
    result["compiler_error_lines"] = compile_.stderr.count("error:")
    if compile_.returncode == 0 and case.get("run", False):
        try:
            run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5,
                                 cwd=out)
            result["run_exit"] = run.returncode
            (out / (tag + "-run.log")).write_text(run.stdout + run.stderr)
        except subprocess.TimeoutExpired:
            result["run_timeout"] = True
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--llama", default="http://127.0.0.1:8200")
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--llama-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = json.loads(Path(args.cases).read_text())
    tok = tokenizer(args.tokenizer)
    template = ChatTemplate(ROOT / "tools/kolibri/chat_template.jinja")
    eng = None if args.llama_only else Engine(
        args.bin, args.model, 32768, args.threads,
        {"STRATA_KOLIBRI_REUSE": "0",
         "STRATA_KOLIBRI_LOGITS_DUMP": str(out / "strata-logits.bin")})
    rows = []
    try:
        for case in cases:
            messages = [{"role": "system", "content": case["system"]},
                        {"role": "user", "content": case["user"]}]
            kwargs = {"enable_thinking": case.get("enable_thinking", False)}
            if case.get("reasoning_effort"):
                kwargs["reasoning_effort"] = case["reasoning_effort"]
            prompt = template.render(messages, **kwargs)
            ids = tok.encode(prompt, parse_special=True)
            llama_ids = post(args.llama + "/tokenize",
                             {"content": prompt, "add_special": False,
                              "parse_special": True})["tokens"]
            if ids != llama_ids:
                raise RuntimeError(f"tokenizer mismatch for {case['tag']}")
            (out / (case["tag"] + "-prompt.json")).write_text(
                json.dumps({"prompt": prompt, "ids": ids, "case": case}, indent=2))
            for name in (("llama",) if args.llama_only else ("llama", "strata")):
                start = time.monotonic()
                if name == "llama":
                    response = post(args.llama + "/completion", {
                        "prompt": ids, "n_predict": case.get("cap", 2048),
                        "temperature": 0, "top_k": 1, "top_p": 1, "min_p": 0,
                        "repeat_penalty": 1, "presence_penalty": 0,
                        "frequency_penalty": 0, "seed": 42,
                        "cache_prompt": False, "return_tokens": True})
                    text = response["content"]
                else:
                    response = eng.send(ids, case.get("cap", 2048))
                    text = tok.decode(response["gen"])
                elapsed = time.monotonic() - start
                tag = case["tag"] + "-" + name
                (out / (tag + "-response.json")).write_text(json.dumps(response, indent=2))
                (out / (tag + ".txt")).write_text(text)
                result = evaluate(out, tag, text, case)
                row = {"tag": case["tag"], "engine": name, "prompt_tokens": len(ids),
                       "seconds": elapsed, **result}
                rows.append(row)
                (out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
                print(json.dumps(row), flush=True)
    finally:
        if eng is not None:
            eng.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
