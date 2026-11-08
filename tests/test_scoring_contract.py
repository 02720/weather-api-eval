"""SCORING.md 与生产权重表的契约测试（发现 9：单一事实源）。

文档、配置、代码、页面、测试必须描述同一套口径。这里只锁最容易腐烂的一处：
SCORING.md 的权重表逐项等于 `TEMP_SKILL_PARTS` / `PRECIP_SKILL_PARTS`。
"""
from __future__ import annotations

import re
from pathlib import Path

from weather_eval import evaluate as ev
from weather_eval.config import DEFAULT_EVAL, load_config

ROOT = Path(__file__).resolve().parents[1]
SCORING = ROOT / "SCORING.md"


def _section(text: str, start: str, end: str) -> str:
    i = text.index(start)
    j = text.index(end, i + len(start))
    return text[i:j]


def _weight_table(section: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in section.splitlines():
        m = re.match(r"^\|\s*([a-z0-9_]+)\s*\|\s*([0-9.]+)\s*\|", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def test_scoring_doc_matches_production_skill_parts():
    text = SCORING.read_text(encoding="utf-8")
    temp = _weight_table(_section(text, "## 2. 温度维", "## 3. 降水维"))
    precip = _weight_table(_section(text, "## 3. 降水维", "## 4. 参照"))
    assert temp == {k: float(w) for k, w, *_ in ev.TEMP_SKILL_PARTS}
    assert precip == {k: float(w) for k, w, *_ in ev.PRECIP_SKILL_PARTS}


def test_scoring_doc_states_current_alpha_and_reference_mode():
    text = SCORING.read_text(encoding="utf-8")
    assert "α = 0.5" in text
    assert "score_ref_mode: fixed" in text
    cfg = load_config(None)
    assert float(cfg.eval["score_composite_alpha"]) == 0.5
    assert str(cfg.eval["score_ref_mode"]) == "fixed"


def test_deprecated_params_are_declared_not_active_knobs():
    cfg = load_config(None)
    declared = set(cfg.eval.get("deprecated_in_skill_v5") or [])
    assert {"score_skill_k", "score_index_gain", "score_composite_gain",
            "score_penalty_lambda_temp", "score_penalty_lambda_precip",
            "score_slopes"} <= declared
    # 退役参数仍在 DEFAULT_EVAL/配置里供归档回放，但 skill-v5 生产路径不读
    # 其中任何一项（grep 级断言由源码导出的口径测试覆盖；这里锁清单本身）。
    assert isinstance(DEFAULT_EVAL["score_skill_k"], float)
    assert isinstance(cfg.eval["score_skill_k"], float)
