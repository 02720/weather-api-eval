# 第八次重冻结（提案）：P1 上端留白 + P3 λ 复标定

本文件是**待执行的第八次 Golden Master 重冻结**的交付包：给中央重冻结的人留一份
可直接粘贴的 `tests/baseline/CHANGELOG.md` 条目、精确的字段差异预测与复现命令。
**本文件不修改 `tests/baseline/` 下任何内容，也不代替重冻结动作。**

- 变更内容：审查 P1（综合分上端饱和）+ P3（λ 契约漂移）合并处置（另有此前已落地、
  尚未重冻结的 P8 降水最小存活权重门槛一并入账）。
- 口径与方法留档：[docs/score_slopes.md §九](score_slopes.md)。
- 冻结轴：沿用当前 `tests/baseline/window.json`（2026-08-01T00:00 ~ 2026-10-14T23:00，
  `n_boot_days=50`），用 `scripts/make_baseline.py` 重新生成。

---

## 一、拟写入 `tests/baseline/CHANGELOG.md` 的条目（置于"第七次"条目之上）

```markdown
## 2026-10-__ · 第八次重冻结（P1 上端留白 + P3 λ 复标定；含 P8 证据门槛）

重冻结原因：**口径变更**（评分体系第一性原理审查 P1/P3 合并处置，另含此前已落地
的 P8）。冻结窗口与输入指纹复用当前轴（2026-08-01 ~ 2026-10-14T23:00）——差异
只可能来自口径。

### 变了什么（三方归因）

**A. 此前已落地的 P8（降水最小存活权重门槛，先于本次入账）**

* 2/7 降水指标存活即重归一化的"伪分"消失：ukmo `hourly:7d` 的伪造 #1（97.4）、
  ncep_hgefs025 `hourly:10d`（92.53）等 8 行由 qualified 翻为不合格（score=None）；
* 冠军换人 5/51 榜：`all:2d`/`daily:2d`/`hourly:2d`（cma_gowfs_v1→msn_v1）、
  `hourly:10d`（ncep_hgefs025→tj_km_fusion）、`hourly:7d`（ukmo→ecmwf_aifs025_single）；
* 逐桶难度对齐的参与集合随之变化：`acc2`/`rmse` 各 54 行、`ts`/`ets` 个别行出现
  ≤1.84 的对齐后位移（ukmo `hourly` 一行最大）。

**B. 本次 P3：λ 全量复标（15 项）**

* 旧 λ 在冻结窗口上 12/15 项实测子分 sd 跌破 8（最低 amt_mae 4.91、slope 5.95）；
* 新 λ 全部复归 sd=8.00±0.01；逐项新旧对照与白话见 docs/score_slopes.md §九 表。

**C. 本次 P1：综合分上端留白（gain 2.66 → 1.851）**

* 旧口径缩放前维度分 >68.80 即 clip 到 100：实测 161/1541 格（10.45%）撞顶、
  12 个桶内 ≥2 格并列（1d/2d 温度维成片）；
* 新规则 gain=(98−50)/(P99.5−50)=1.851：撞顶 2/1541（0.13%）、无同桶并列
  （cma_gowfs_v1 日降水 1d 的 89.69 极值格、fuxi_c88 日温度 1d min 的 78.13 格）。

### 效果与名次影响（前后不可直接对比）

| 榜 | 综合分极差（base→new） | sd | 中位 CI90 宽 | 冠军 | Spearman（vs 第七次基线） |
|---|---|---|---|---|---|
| 总榜 all | 25.48 → **21.44** | 6.82 → 5.76 | 17.6 → 15.9 | fuxi_c88（不变） | 0.9803 |
| 小时榜 hourly | 31.49 → **25.63** | 7.94 → 6.52 | 16.8 → 15.3 | ecmwf_aifs025_single（不变） | 0.9650 |
| 日榜 daily | 19.60 → **17.06** | 5.54 → 4.85 | 19.2 → 16.7 | fuxi_c88（不变） | 0.9855 |

* 51 张榜中 47 张有名次位移（合计 535 个位置）；子榜冠军换人 9 张 = P8 5 张 + 本次 4 张
  （`all:14d`、`all:7d` 归 fuxi_c88；`hourly:1d` cma_gowfs_v1→msn_v1、
  `hourly:4d` msn_v1→ecmwf_aifs025_single）；
* `sig_vs_top` 3 行 False→True（all/geovis_v1、daily/fengwu_ghr_9km、daily/tj_t2_early）——
  λ 变更后区间与头名的分离度变化，非显示层效应；
* 三个主榜冠军不变；全 47 张（n≥8）子榜 Spearman 中位 0.983、最小 0.891。

### 凭什么认为"其它字段没变"

* `scorecard` 逐位不变（**0/1740**）；
* 行级计数全部逐位不变：`n` / `n_precip` / `n_eff` / `n_eff_rain` / `lead_days` /
  `n_days` / `n_days_rain` / `n_daily_days`（按 model 配对的最大差 = 0）；
* 窗口与输入指纹复用当前轴（`tests/baseline/window.json` 原样），观测指纹未动；
* 逐字段差异预测（按 model 配对）：`temp_score` 1004 行、`score` 977 行、
  `precip_score` 969 行、`ci90` 81 行（三个主榜全行）、`acc2`/`rmse` 各 54 行、
  `daily_score`/`hourly_score`/`track_gap` 各 27 行、`champion_pct` 17 行、
  `qualified` 8 行、`sig_vs_top` 3 行。

### 附注（文档一致性，供重冻结顺手处理）

* 第七次条目写"冻结窗口 2026-08-01 ~ 2026-10-04（--end 显式指定）"，而当前
  `window.json` 已是 2026-10-14（更早的 `9d283403` 重冻结把窗口扩到 10-14，未留
  CHANGELOG 条目）——建议在第八次条目里补一句说明，或修订第七次条目的窗口措辞。
* 第七次条目"当前数据下没有任何展示值触及 100 上限（971 个分数最大 97.40）"是
  **10-04 轴**的结论；窗口扩到 10-14 后缩放前维度分撞顶才显形（161/1541 格），
  两条结论各自成立、不矛盾。
```

---

## 二、精确字段差异预测（按 model 配对；重冻结后自检用）

数据：`tests/baseline/report_baseline.json`（当前 HEAD 基线）对拍
`/tmp/opencode/p13/new.json`（当前工作树口径、同窗口）。重冻结后若字段集合与
量级与下表不符，应先查"是不是跑的不是同一轴"再谈。

| 字段 | 变更行数（51 榜×27~29 行） | 最大 \|Δ\| | 归因 |
|---|---:|---:|---|
| `temp_score` | 1004 | 13.25 | λ+gain（P1/P3） |
| `score` | 977 | 21.08 | λ+gain（P1/P3） |
| `precip_score` | 969 | 29.17 | λ+gain + P8 门槛 |
| `ci90` | 81 | — | gain 缩放 |
| `acc2` / `rmse` | 54 / 54 | 1.838 / 0.085 | P8 对齐集合位移 |
| `daily_score` / `hourly_score` | 27 / 27 | 12.37 / 12.90 | λ+gain |
| `track_gap` | 27 | 3.25 | 同上 |
| `champion_pct` | 17 | 6.40 | λ（自举冠军频率） |
| `qualified` | 8 | — | P8（伪证据行出清） |
| `sig_vs_top` | 3 | — | λ |
| `n` * `n_precip` * `n_eff` * `n_eff_rain` * `lead_days` * `n_days` * `n_days_rain` * `n_daily_days` | **0** | 0 | 计数不受口径影响 |
| `scorecard`（全部 1740 字段） | **0** | — | 逐位不变 |

就绪状态：`test_report_matches_frozen_baseline` 当前红灯（预期），其余 golden 守卫
（`test_baseline_files_exist` / `test_window_matches_frozen_days` /
`test_window_is_settled` / `test_frozen_inputs_unchanged`）/ 性能预算全绿。

## 三、复现命令

```bash
# 1) 标定（冻结窗口捕获；含 §1.5 上端门、§1.6 留白扫描、gain 反解）
PYTHONPATH=src python scripts/calibrate_score_slopes.py \
    --from-json .work/track_sources_baseline_window.json

# 2) 全量 A/B（pre=旧 λ+2.66 → new=新 λ+1.851；冻结窗口；约 2 分钟）
python /tmp/opencode/p13/run_ab.py          # 产出 /tmp/opencode/p13/{pre,new}.json
python /tmp/opencode/p13/delta.py           # 产出 /tmp/opencode/p13/delta.md（名次台账）
python /tmp/opencode/p13/baseline_delta.py  # 产出 /tmp/opencode/p13/baseline_delta.json

# 3) 中央重冻结（由重冻结执行人运行；本次交付不运行）
python scripts/make_baseline.py
```

注：`/tmp/opencode/p13/` 为本机临时交付目录（run_ab.py / delta.py / baseline_delta.py /
pre.json / new.json / delta.md / baseline_delta.json / cal_p1p3.log）。其中全量 A/B
依赖 `.work/track_sources_baseline_window.json` 与现行配置；若目录已清理，
按第 1 步重建捕获后重跑第 2 步即可。
