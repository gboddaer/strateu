"""Check the diagnostic instruments independently of model-generated answers."""
import json
import math
from pathlib import Path
import shutil
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from code_quality_probe import evaluate, source_of
from code_quality_distributions import last_logits, logsoftmax


def test_reasoning_code_is_not_mistaken_for_the_answer():
    text = '<think>```cpp\nint wrong;\n```</think>\n```cpp\nint correct;\n```'
    assert source_of(text) == 'int correct;\n'


def test_unclosed_thinking_is_no_answer_not_a_compile_failure(tmp_path):
    result = evaluate(tmp_path, 'unfinished', '<think>```cpp\nint main(){}\n```',
                      {'enable_thinking': True})
    assert result == {'no_answer': True, 'reasoning_closed': False}
    assert not (tmp_path / 'unfinished.cpp').exists()


def test_logsoftmax_matches_hand_calculated_two_to_one_odds():
    actual = logsoftmax(np.array([math.log(2), 0]))
    assert actual == pytest.approx([math.log(2 / 3), math.log(1 / 3)])


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_distribution_rejects_nonfinite_scores(value):
    with pytest.raises(ValueError, match='non-finite'):
        logsoftmax([0, value])


def test_reads_the_final_finite_vector_not_the_first(tmp_path):
    path = tmp_path / 'logits.bin'
    path.write_bytes(struct.pack('=iii2f', 0, 1, 2, 1, 2)
                     + struct.pack('=iii2f', 0, 2, 2, 3, 4))
    assert last_logits(path).tolist() == [3, 4]


def test_rejects_truncated_logits_record(tmp_path):
    path = tmp_path / 'logits.bin'
    path.write_bytes(struct.pack('=iii', 0, 1, 2) + b'\0' * 4)
    with pytest.raises(ValueError, match='invalid logits'):
        last_logits(path)


@pytest.mark.skipif(sys.platform != 'linux' or shutil.which('g++') is None,
                    reason='Linux C++ oracle calibration needs g++')
@pytest.mark.parametrize('tag', ['insert-char', 'line-edit', 'raw-mode', 'nonblocking-read'])
def test_coding_oracle_accepts_known_correct_reference(tmp_path, tag):
    cases = json.loads(Path(__file__).with_name('code_quality_cases.json').read_text())
    case = next(c for c in cases if c['tag'] == tag)
    result = evaluate(tmp_path, tag, case['reference'], case)
    assert result['compile_exit'] == 0
    assert result['run_exit'] == 0


@pytest.mark.skipif(sys.platform != 'linux' or shutil.which('g++') is None,
                    reason='Linux PTY calibration needs g++')
def test_raw_mode_oracle_accepts_disabling_flow_control_characters(tmp_path):
    cases = json.loads(Path(__file__).with_name('code_quality_cases.json').read_text())
    case = next(c for c in cases if c['tag'] == 'raw-mode')
    # On Linux, zero disables these control characters. Clearing IXON is not
    # the only implementation that delivers Ctrl+S and Ctrl+Q as input bytes.
    source = '''termios editor_raw(termios a) {
        a.c_lflag &= ~(ICANON|ECHO|ISIG);
        a.c_cc[VMIN]=1; a.c_cc[VTIME]=0;
        a.c_cc[VSTART]=0; a.c_cc[VSTOP]=0;
        return a;
    }'''
    result = evaluate(tmp_path, 'alternative', source, case)
    assert result['compile_exit'] == 0
    assert result['run_exit'] == 0


@pytest.mark.skipif(sys.platform != 'linux' or shutil.which('g++') is None,
                    reason='Linux PTY calibration needs g++')
def test_raw_mode_oracle_rejects_intercepting_save_and_quit(tmp_path):
    cases = json.loads(Path(__file__).with_name('code_quality_cases.json').read_text())
    case = next(c for c in cases if c['tag'] == 'raw-mode')
    source = '''termios editor_raw(termios a) {
        a.c_lflag &= ~(ICANON|ECHO|ISIG);
        a.c_cc[VMIN]=1; a.c_cc[VTIME]=0;
        a.c_cc[VSTART]=17; a.c_cc[VSTOP]=19;
        return a;
    }'''
    result = evaluate(tmp_path, 'intercepted', source, case)
    assert result['compile_exit'] == 0
    assert result['run_exit'] != 0
