"""The JSON call form: `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`.

Kolibri-1's own chat template (tools/kolibri/chat_template.jinja, the one its tokenizer pack ships) tells the model to
write its calls in exactly this form.  `OutputParser` reads the XML `<function=...>` block first and the JSON form only
under `tool_call_recovery`, which no launch path set - so the model's calls came back as the answer's text and an agent
client saw `<tool_call>{"name": "read", ...}</tool_call>` instead of a tool call it could run.

The default now follows the template: a template that asks for the JSON form has its calls read as calls, and every
other template keeps the default (off, the XML form) unchanged.
"""
from __future__ import annotations

import json
import sys
import threading
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve.frontend as frontend  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve, tool_call_recovery_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KOLIBRI = ROOT / "tools/kolibri/chat_template.jinja"
QWEN = ROOT / "serve/chat_template.jinja"
DEEPSEEK = ROOT / "serve/fixtures/deepseek41/chat_template.jinja"
CTX = 4096

PROPS = {"path": {"type": "string"}, "city": {"type": "string"}, "content": {"type": "string"}}
TOOLS = [{"name": "read", "description": "Read a file", "parameters": {"type": "object", "properties": PROPS}},
         {"name": "get_weather", "description": "Weather for a city",
          "parameters": {"type": "object", "properties": PROPS}}]

# the shape a client reported from a served Kolibri-1 (three calls in one answer, arguments spread over lines)
REPORTED = ('<tool_call>\n{"name": "read", "arguments": {"path": "/a/SKILL.md"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "read", "arguments": {"path": "/b/KOLIBRI1.md"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "read", "arguments": {"path": "/c/AMD_HIP.md"}}\n</tool_call>')
REPORTED_CALLS = [("read", {"path": "/a/SKILL.md"}), ("read", {"path": "/b/KOLIBRI1.md"}),
                  ("read", {"path": "/c/AMD_HIP.md"})]


def parse(text, width=0, thinking=False, **kw):
    """-> (tool calls, content text, parser); `width` 0 feeds the text whole, else that many characters at a time."""
    p = frontend.OutputParser(thinking=thinking, stream_tools=True, **kw)
    evs = []
    for k in range(0, len(text), width) if width else [0]:
        evs += p.feed(text[k:k + width] if width else text)
    evs += p.finish()
    return ([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"],
            "".join(e.text or "" for e in evs if e.kind == "content").strip(), p)


class TemplateForm(unittest.TestCase):
    """Which call form a template asks for - the rule the default follows."""

    def test_the_kolibri_template_asks_for_the_json_form(self):
        self.assertTrue(frontend.template_json_calls(ChatTemplate(KOLIBRI)))

    def test_the_xml_templates_do_not(self):
        for path in (QWEN, DEEPSEEK, ROOT / "serve/chat_template.jinja"):
            with self.subTest(path=path.name):
                self.assertFalse(frontend.template_json_calls(ChatTemplate(path)))


class RecoveryDefault(unittest.TestCase):
    """The config's switch still wins; unset, the template decides.  Every other family keeps today's default."""

    def test_unset_follows_the_template(self):
        self.assertTrue(tool_call_recovery_of({}, ChatTemplate(KOLIBRI)))
        for cfg in ({}, {"model_name": "qwen3.8-flash-next"}):
            self.assertFalse(tool_call_recovery_of(cfg, ChatTemplate(QWEN)))

    def test_the_config_wins_both_ways(self):
        self.assertFalse(tool_call_recovery_of({"tool_call_recovery": False}, ChatTemplate(KOLIBRI)))
        self.assertTrue(tool_call_recovery_of({"tool_call_recovery": True}, ChatTemplate(QWEN)))

    def test_a_non_boolean_is_refused(self):
        for bad in ("yes", "true", 1, []):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    tool_call_recovery_of({"tool_call_recovery": bad}, ChatTemplate(QWEN))


class JsonFormIsACall(unittest.TestCase):
    """The parser reads the JSON form as calls once the switch is on - the model's own spelling, as Kolibri writes it."""

    def calls(self, text, width=0, **kw):
        return parse(text, width, tools=TOOLS, recover=True, **kw)

    def test_the_reported_answer_is_three_calls(self):
        for width in (0, 1, 3, 17, 10_000):
            with self.subTest(width=width):
                calls, text, p = self.calls(REPORTED, width)
                self.assertEqual(calls, REPORTED_CALLS)
                self.assertEqual(text, "")
                self.assertEqual(p.unread, 0)

    def test_arguments_over_several_lines(self):
        calls, text, _ = self.calls('<tool_call>\n{"name": "read", "arguments": {\n  "path": "/a.md"\n}}\n</tool_call>')
        self.assertEqual((calls, text), ([("read", {"path": "/a.md"})], ""))

    def test_arguments_as_a_json_string_and_under_parameters(self):
        for body in ('{"name": "read", "arguments": "{\\"path\\": \\"/a.md\\"}"}',
                     '{"name": "read", "parameters": {"path": "/a.md"}}'):
            with self.subTest(body=body[:30]):
                self.assertEqual(self.calls(f"<tool_call>\n{body}\n</tool_call>")[0], [("read", {"path": "/a.md"})])

    def test_text_before_and_after_the_calls_survives(self):
        calls, text, _ = self.calls("Reading the three files.\n" + REPORTED + "\nDone.")
        self.assertEqual(calls, REPORTED_CALLS)
        self.assertEqual(text, "Reading the three files.Done.")     # the newline right after a call is dropped, as always

    def test_a_call_for_a_tool_the_request_did_not_declare_stays_text_and_is_counted(self):
        raw = '<tool_call>\n{"name": "format_disk", "arguments": {"path": "/dev/sda"}}\n</tool_call>'
        calls, text, p = self.calls(raw)
        self.assertEqual((calls, text), ([], raw))
        self.assertEqual((p.unread, p.unread_names), (1, ["format_disk"]))


class JsonFormWithTheSwitchOff(unittest.TestCase):
    """Today's default: the block is text.  It is now counted, so the server's log and /metrics say it happened."""

    def test_the_block_goes_back_as_text_and_is_counted(self):
        raw = REPORTED
        for width in (0, 1, 40):
            with self.subTest(width=width):
                calls, text, p = parse(raw, width, tools=TOOLS)
                self.assertEqual((calls, text), ([], raw))
                self.assertGreaterEqual(p.unread, 1)


class ServedReply(unittest.TestCase):
    """What a client sees, end to end: the pack's own template, no config, and the answer's JSON calls arrive as
    `tool_calls` with `finish_reason: "tool_calls"`."""

    def reply(self, template, stream, script="\n\n" + REPORTED, cfg=None):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, script, max_context=CTX), tok, ChatTemplate(template))
        svc.tool_call_recovery = tool_call_recovery_of(cfg or {}, svc.template)
        httpd = serve(svc, port=0)
        try:
            body = {"model": svc.model, "stream": stream, "max_tokens": 300, "messages": [
                        {"role": "user", "content": "Read the three files."}],
                    "tools": [{"type": "function", "function": t} for t in TOOLS]}
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions",
                                        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as r:
                raw = r.read().decode()
        finally:
            httpd.shutdown()
            httpd.server_close()
        if not stream:
            choice = json.loads(raw)["choices"][0]
            return choice["finish_reason"], [(c["function"]["name"], json.loads(c["function"]["arguments"] or "{}"))
                                             for c in choice["message"].get("tool_calls") or []]
        finish, calls, pieces = None, [], {}
        for line in raw.splitlines():
            if not line.startswith("data: {"):
                continue
            chunk = json.loads(line[6:])
            choice = chunk["choices"][0]
            finish = choice.get("finish_reason") or finish
            for c in choice["delta"].get("tool_calls") or []:
                at = c.get("index", 0)
                pieces.setdefault(at, {"name": "", "arguments": ""})
                pieces[at]["name"] = c.get("function", {}).get("name") or pieces[at]["name"]
                pieces[at]["arguments"] += (c.get("function") or {}).get("arguments") or ""
        calls = [(v["name"], json.loads(v["arguments"] or "{}")) for _, v in sorted(pieces.items())]
        return finish, calls

    def test_the_kolibri_template_needs_no_config(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.assertEqual(self.reply(KOLIBRI, stream), ("tool_calls", REPORTED_CALLS))

    def test_a_json_call_inside_the_thinking_is_a_call_too(self):
        """Kolibri thinks before it calls: a JSON block the thinking ends on is the call (the XML form always was)."""
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.assertEqual(self.reply(KOLIBRI, stream, "We need three files.\n" + REPORTED + "\n"),
                                 ("tool_calls", REPORTED_CALLS))

    def test_the_switch_off_keeps_the_text(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                finish, calls = self.reply(KOLIBRI, stream, cfg={"tool_call_recovery": False})
                self.assertEqual((finish, calls), ("stop", []))

    def test_an_xml_calling_template_is_unchanged(self):
        """The same model output under the shipped Qwen template: text, as before (that template asks for `<function=`)."""
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.assertEqual(self.reply(QWEN, stream), ("stop", []))


if __name__ == "__main__":
    unittest.main()
