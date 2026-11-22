"""录制并写入名次分 R 的**冻结尺度**（skill-v6 D4 的接线动作）。

背景（第一性原理）：`R = 50 + scale·√ρ_c·(α z_T+(1−α) z_P)/σ_d` 里的 z 分
分母是**当期参评池**的离散度。只要分母来自当期数据，增删一个源就必然移动
所有幸存者的分数——实测加入一个源可让幸存者 R 变 7.0 分。这由 z 分的定义
决定，不是精度问题，也不是靠 Winsor 能消除的（Winsor 只削弱离群源的影响）。

`stats.rank_from_scores(frozen_scale=...)` 已实现冻结分支，并且**逐位可测**：
给同一份 `{mu_t, sd_t, mu_p, sd_p, phi}`，加入/删除任意源后幸存者 `|ΔR|=0`
（`tests/test_skill_v6.py::test_v6_frozen_scale_makes_rank_pool_independent`）。

本脚本把标定窗口上录到的逐列尺度写进 `config/stations.yaml` 的
`eval.score_rank_frozen_scale`，格式**按榜分轨**：

    score_rank_frozen_scale:
      hourly: {mu_t: [...], sd_t: [...], mu_p: [...], sd_p: [...], phi: [...]}
      daily:  {...}

总榜（`all`）不单独存：它是 hourly+daily 的**横向拼接**，读取时按段拼
（`evaluate._frozen_scale_for`），列序与 `_panel_matrices` 的 hstack 严格一致。

用法：
    # 只看摘要（缺省，不写盘）
    PYTHONPATH=src .venv312/bin/python scripts/freeze_rank_scale.py
    # 写入 config/stations.yaml
    PYTHONPATH=src .venv312/bin/python scripts/freeze_rank_scale.py --write

**写入后必须做的事**（脚本结尾会打印提醒）：
  1. 重冻结 Golden Master 基线（`scripts/make_baseline.py`）并在
     `tests/baseline/CHANGELOG.md` 记一笔——冻结尺度改变 R 的数值。
  2. 重跑 `python -m weather_eval report` 让页面重新发布。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np

from weather_eval import evaluate as ev
from weather_eval.config import load_config
from weather_eval.timeutil import parse_iso

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "stations.yaml"
_KEYS = ("mu_t", "sd_t", "mu_p", "sd_p", "phi")


def record(eval_cfg: dict, start, end, period_label: str) -> dict:
    """跑一次报告，取回逐轨道的逐列 μ/σ/φ。

    录制时必须**用当期池**（先把冻结尺度置空），否则会把自己冻结的值再录一遍。
    """
    cfg = load_config(None)
    eval_cfg = dict(eval_cfg)
    eval_cfg["score_rank_frozen_scale"] = None
    ev.apply_frozen_rank_scale(None)
    report = ev.build_report(cfg.station_ids, cfg.models, eval_cfg, start, end,
                             period_label=period_label)
    return (report.get("meta") or {}).get("rank_scale_diagnostics") or {}


def _summarise(values) -> dict:
    arr = np.asarray([v for v in (values or []) if v is not None], dtype=float)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "median": round(float(np.median(arr)), 4),
        "iqr": round(float(np.percentile(arr, 75) - np.percentile(arr, 25)), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
    }


def _clean_scale(diag: dict) -> dict:
    """只保留两轨、丢掉 None 列（None = 该列不可估，读取时按列数校验会拒绝）。"""
    out: dict[str, dict[str, list[float]]] = {}
    for track in ("hourly", "daily"):
        cols = diag.get(track) or {}
        if not cols:
            continue
        out[track] = {}
        for key in _KEYS:
            vals = cols.get(key) or []
            # None 用该轨中位数回填：整列缺尺度意味着"这一列不可估"，
            # 回填中位数比留空洞更诚实（留洞会让整份尺度因列数不符而失效）。
            arr = np.asarray([np.nan if v is None else float(v) for v in vals],
                             dtype=float)
            finite = np.isfinite(arr)
            if not finite.any():
                raise SystemExit(f"{track}/{key} 整列不可估，拒绝写入"
                                 "（请扩大标定窗口）")
            if not finite.all():
                arr[~finite] = float(np.median(arr[finite]))
            out[track][key] = [round(float(v), 6) for v in arr]
    return out


def _render_yaml_block(scale: dict) -> str:
    lines = ["  score_rank_frozen_scale:"]
    for track in ("hourly", "daily"):
        cols = scale[track]
        lines.append(f"    {track}:")
        for key in _KEYS:
            vals = ", ".join(f"{v}" for v in cols[key])
            lines.append(f"      {key}: [{vals}]")
    return "\n".join(lines) + "\n"


def _write_config(scale: dict) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    block = _render_yaml_block(scale)
    pat = re.compile(
        r"^  score_rank_frozen_scale:.*?(?=^  [a-zA-Z_]|\Z)", re.S | re.M)
    if pat.search(text):
        text = pat.sub(block, text, count=1)
    else:
        # 插在 score_scheme 行之后，保持相邻性便于人工核对
        anchor = re.search(r"^  score_scheme:.*$", text, re.M)
        if not anchor:
            raise SystemExit("config/stations.yaml 里找不到 score_scheme 行")
        text = (text[:anchor.end()] + "\n" + block +
                text[anchor.end():])
    CONFIG_PATH.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window", default="tests/baseline/window.json",
                    help="标定窗口 JSON（含 start_dt/end_dt/period_label）")
    ap.add_argument("--write", action="store_true",
                    help="写入 config/stations.yaml（缺省只打印摘要）")
    args = ap.parse_args(argv)

    win = json.loads(Path(args.window).read_text(encoding="utf-8"))
    cfg = load_config(None)
    diag = record(cfg.eval, parse_iso(win["start_dt"]), parse_iso(win["end_dt"]),
                  win.get("period_label", ""))
    if not diag:
        print("未取到 rank_scale_diagnostics：本版 build_report 不发布该字段，"
              "或已处于冻结状态。请先置空 eval.score_rank_frozen_scale。",
              file=sys.stderr)
        return 2

    print(f"标定窗口：{win['start_dt']} ~ {win['end_dt']}")
    for track in ("hourly", "daily"):
        cols = diag.get(track) or {}
        if not cols:
            continue
        n_cols = max(len(v or []) for v in cols.values())
        print(f"[{track}] 列数 = {n_cols}")
        for key in _KEYS:
            print(f"   {key:5s} {_summarise(cols.get(key))}")

    scale = _clean_scale(diag)
    for track, cols in scale.items():
        n = len(cols["mu_t"])
        for key in _KEYS:
            assert len(cols[key]) == n, f"{track}/{key} 列数不一致"
    print("\n两轨列数：" + ", ".join(
        f"{t}={len(c['mu_t'])}" for t, c in scale.items()))
    print("（总榜 all 不单独存：它是两轨的横向拼接，读取时按段组装。）")

    if not args.write:
        print("\n未写入（加 --write 落盘到 config/stations.yaml）。")
        return 0

    _write_config(scale)
    print(f"\n已写入 {CONFIG_PATH.relative_to(PROJECT_ROOT)} 的 "
          "eval.score_rank_frozen_scale。")
    print("下一步必须做（否则基线/页面与配置不一致）：")
    print("  1. .venv312/bin/python scripts/make_baseline.py   # 重冻结基线")
    print("  2. 在 tests/baseline/CHANGELOG.md 记一笔")
    print("  3. PYTHONPATH=src .venv312/bin/python -m weather_eval report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
