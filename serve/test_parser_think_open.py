"""The opening `\u003cthink\u003e` the model writes itself is a control tag, not reasoning text.

Kolibri-1 emits `\u003cthink\u003e` as its first token (measured: logprob -3.8e-06 at temperature 0 through both this server
and llama.cpp), and llama.cpp's parser drops it.  A client that shows `reasoning_content` must not see the marker.
"""
import unittest

from serve.frontend import OutputParser, THINK_START, THINK_END


def reasoning(events):
    return "".join(e.text or "" for e in events if e.kind == "reasoning")


def content(events):
    return "".join(e.text or "" for e in events if e.kind == "content")


class ThinkOpenMarker(unittest.TestCase):
    def test_the_marker_is_not_part_of_the_reasoning(self):
        p = OutputParser(thinking=True)
        got = p.feed(THINK_START + "\n1.  Analyze the request.\n")
        self.assertEqual(reasoning(got), "1.  Analyze the request.\n")

    def test_a_marker_split_across_deltas_is_held_back(self):
        p = OutputParser(thinking=True)
        got = []
        pieces = (THINK_START[:3], THINK_START[3:6], THINK_START[6:], "\n", "first ", "thought")
        for piece in pieces:
            got += p.feed(piece)
        self.assertEqual(reasoning(got), "first thought")

    def test_a_reply_that_opens_on_its_text_is_unchanged(self):
        p = OutputParser(thinking=True)
        self.assertEqual(reasoning(p.feed("first thought")), "first thought")

    def test_an_empty_thinking_still_opens_the_answer(self):
        p = OutputParser(thinking=True)
        got = p.feed(THINK_START + "\n" + THINK_END + "\n\nThe answer is 4.")
        self.assertNotIn(THINK_START, reasoning(got))
        self.assertEqual(content(got).strip(), "The answer is 4.")

    def test_a_partial_marker_at_the_end_is_not_lost(self):
        p = OutputParser(thinking=True)
        got = p.feed(THINK_START[:4])
        got += p.finish("stop")
        self.assertTrue((reasoning(got) + content(got)).strip(), "the held text must still be sent")

    def test_a_marker_after_the_first_text_is_left_alone(self):
        p = OutputParser(thinking=True)
        got = p.feed("quote: ") + p.feed(THINK_START + "\n")
        self.assertIn(THINK_START, reasoning(got))     # a quoted tag further in is the model's own text

    def test_without_thinking_the_marker_is_plain_text(self):
        p = OutputParser(thinking=False)
        self.assertEqual(content(p.feed(THINK_START + "\nhello")), THINK_START + "\nhello")


if __name__ == "__main__":
    unittest.main()
