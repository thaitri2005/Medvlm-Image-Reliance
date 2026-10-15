import io

from medvlm.logs import _Tee
from medvlm.models.base import Output
from medvlm.run import _fmt_dur, _Progress


def test_tee_keeps_terminal_output_and_writes_clean_timestamped_lines():
    term, log = io.StringIO(), io.StringIO()
    t = _Tee(term, log, "[stderr] ")
    t.write("hello\n")
    t.write("\r 10%|#  |\r 50%|##  |\r100%|####|")   # a progress bar redrawing in place
    t.write("\npart")
    t.write("ial line\n\n")
    assert term.getvalue() == "hello\n\r 10%|#  |\r 50%|##  |\r100%|####|\npartial line\n\n"
    lines = log.getvalue().splitlines()
    assert [ln[9:] for ln in lines] == ["[stderr] hello", "[stderr] 100%|####|", "[stderr] partial line"]
    assert all(ln[2] == ":" and ln[5] == ":" for ln in lines)   # HH:MM:SS prefix


def test_fmt_dur():
    assert _fmt_dur(59) == "0m59s"
    assert _fmt_dur(3725) == "1h02m"


class _J:
    def __init__(self, key):
        self.key = key


def test_progress_flags_low_yn_mass_and_empty_open(capsys):
    p = _Progress(total=4, every_sec=1e9)
    outs = [Output(text="yes", p_yes=0.9, yn_mass=0.99), Output(text="maybe", p_yes=0.5, yn_mass=0.4),
            Output(text="", p_yes=None, yn_mass=None), Output(text="liver", p_yes=None, yn_mass=None)]
    p.add([_J(f"s{i}|orig") for i in range(4)], outs)
    out = capsys.readouterr().out
    assert "[warn] s1|orig: yn_mass 0.400" in out
    assert "[progress] 4/4 (100.0%)" in out and "empty open 1/4" in out
    assert p.low_mass_total == 1 and p.empty_total == 1
