"""Tests for .claude/hooks/chat_voice.py, the Stop hook that logs the tells of a chat reply.

Standard library unittest. Run from the repository root:

    python -m unittest discover -s .claude/hooks/tests -v

Each test that writes the log writes it to a temporary folder, so no file is written into the
working tree.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

HOOKS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOOKS))
import chat_voice as cv  # noqa: E402

SC = cv.load_style_check()


def rules(text: str) -> set[str]:
    """The rules the hook finds in a reply."""
    return {r for r, _ in cv.check(text, SC)}


class Tells(unittest.TestCase):
    def test_style_check_loads(self):
        self.assertIsNotNone(SC)
        self.assertTrue(hasattr(SC, "pair_tells"))

    def test_contrast_in_one_sentence(self):
        found = rules("It is not a price signal, it is a cost allocation.")
        self.assertIn("flipped or split negative contrast", found)

    def test_contrast_any_subject(self):
        found = rules("The rule isn't about prices; it's about who pays the rent.")
        self.assertIn("negation corrected in the same sentence (any subject)", found)

    def test_contrast_over_two_sentences(self):
        found = rules("The rent isn't a windfall. It's a payment for the right.")
        self.assertIn(SC.PAIR_TELL, found)

    def test_plain_reply_is_clean(self):
        text = (
            "Line 6-7 binds at its 40 MW rating, with a shadow price of minus 20.10 dollars per MWh.\n\n"
            "The nodal price at bus 7, λ_7, is 37.13 dollars per MWh."
        )
        self.assertEqual(rules(text), set())

    def test_latex_is_flagged(self):
        self.assertIn(
            "chat LaTeX math", rules("The price is $\\lambda_n + \\mu_m$ at bus 7.")
        )
        self.assertIn("chat LaTeX math", rules("The dual is \\mu_m on the line."))

    def test_unicode_math_is_not_latex(self):
        self.assertNotIn(
            "chat LaTeX math",
            rules("The nodal price is λ̂_a + Σ_m SF_{n,m} μ_m at bus n."),
        )

    def test_dollar_amounts_are_not_latex(self):
        self.assertNotIn(
            "chat LaTeX math",
            rules("Load pays $9,745/h, and generation is paid $8,941/h."),
        )

    def test_drama_is_flagged_and_you_is_not(self):
        self.assertIn(
            "chat reader address", rules("Notice that the price at bus 7 jumps.")
        )
        self.assertEqual(
            rules("The commit waits on you, and the choice of hub is yours."), set()
        )

    def test_permission_question_is_flagged(self):
        self.assertIn(
            "chat permission question", rules("Which option should I take, or is there a third?")
        )
        self.assertIn("chat permission question", rules("Should I hold the commit?"))
        self.assertEqual(rules("Proceed with the first option?"), set())
        self.assertEqual(rules("Do you want to proceed with the first option?"), set())

    def test_preemptive_negation_is_flagged(self):
        self.assertIn(
            "preemptive negation", rules("Yes, in three places, and none of them blocks a reply.")
        )
        self.assertIn("preemptive negation", rules("I have not edited the file."))
        self.assertEqual(rules("Each check reports after the text is written."), set())

    def test_em_dash_is_flagged(self):
        self.assertIn(
            "chat em dash", rules("The line binds — and the prices separate.")
        )

    def test_code_and_tables_are_skipped(self):
        text = (
            "The run printed the tally.\n\n"
            "```\nIt is not a test, it is a fixture.\n$\\lambda$\n```\n\n"
            "| case | note |\n|---|---|\n| one | It is not X, it is Y. |\n\n"
            "Inline `\\frac{a}{b}` stays code."
        )
        self.assertEqual(rules(text), set())


class Reading(unittest.TestCase):
    def test_payload_field(self):
        self.assertEqual(cv.reply_text({"last_assistant_message": "Done."}), "Done.")

    def test_transcript_after_last_prompt(self):
        lines = [
            {"type": "user", "message": {"content": "first question"}},
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "old reply"}]},
            },
            {"type": "user", "message": {"content": "second question"}},
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "part one"}]},
            },
            {
                "type": "user",
                "message": {"content": [{"type": "tool_result", "content": "ok"}]},
            },
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "part two"}]},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.jsonl"
            path.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
            self.assertEqual(
                cv.reply_text({"transcript_path": str(path)}), "part one\n\npart two"
            )


class Logging(unittest.TestCase):
    def test_hits_logged_as_voice(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = {
                "session_id": "s1",
                "last_assistant_message": "It is not a tariff, it is a rule.",
            }
            self.assertEqual(cv.main(json.dumps(payload), log_dir=Path(tmp)), 0)
            recs = [
                json.loads(x)
                for x in (Path(tmp) / "hooks.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertTrue(recs)
            self.assertTrue(
                all(r["kind"] == "voice" and r["hook"] == "chat_voice" for r in recs)
            )
            self.assertEqual(recs[0]["session"], "s1")

    def test_clean_reply_logs_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            cv.main(
                json.dumps(
                    {"last_assistant_message": "The suite ran, and 78 tests passed."}
                ),
                log_dir=Path(tmp),
            )
            self.assertFalse((Path(tmp) / "hooks.jsonl").exists())

    def test_bad_input_fails_open(self):
        self.assertEqual(cv.main("not json"), 0)
        self.assertEqual(cv.main(""), 0)


if __name__ == "__main__":
    unittest.main()
