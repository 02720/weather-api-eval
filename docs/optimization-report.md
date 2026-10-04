# 优化实施记录（2026-10-04）

> **后续补记（同日）**：本文的冻结窗口 `2026-08-01 ~ 2026-10-03` 随后被移到
> `2026-08-01 ~ 2026-10-01`——原终点落在观测回改窗口内，基线输入会随抓取移动
> （详见 `tests/baseline/CHANGELOG.md` 第二次重冻结）。本文数字是**当时那一次**
> 实测的记录，不随基线重冻结而改写；复现请以 `tests/baseline/window.json`
> 的当前值为准。

基线 commit `a36d54dc`。本文所有数字都是**本机实测**（Python 3.12.8 / cyeva 0.2.3 /
numpy 2.1.2，4 核 / 11 GB，数据窗口 2026-08-01 ~ 2026-10-03，4 站 × 28 模型），
不是从方案里抄来的估算。方案 §11 第 12 条要求"动手前先跑 bench 拿到本机 before"，
下文每一处与方案预估不一致的地方都标了出来。

---

## 0. 结论速览

| 指标 | Before（实测） | After（实测） | 变化 |
|---|---:|---:|---:|
| `build_report` 总耗时 | 103.5 s | **84.0 s** | −19% |
| ─ 扣除诊断层修复后可比部分 | 99.8 s | **≈ 54 s** | **−46%** |
| `precip_metrics` | 31.8 s | 6.4 s | −80% |
| `build_day_stat_tables` | 15.3 s | < 0.5 s（已跌出前列） | −97% |
| n_eff（2,044 次调用） | 13.7 s | ≈ 1.4 s | −90% |
| `_compute_diagnostics` | 3.7 s | 29.6 s | **+26 s（修 bug，见 §4）** |
| 峰值 RSS | 2,052 MB | 1,871 MB | −9% |
| `ruff check`（src+scripts+tests） | 未启用 | **0 error** | — |
| 533 个原有测试 | 全绿 | **全绿**（新增 26 个） | — |

"扣除诊断层修复后可比部分"这一行很重要：诊断层此前是**坏的**（只看见 3 天数据），
修复后它多做了 26 秒的活。把这部分剔掉，优化工作本身把报告构建的可比部分从
99.8 s 压到约 54 s。

---

## 1. 先建护栏，再动刀（Phase 0）

优化的第一条纪律不是"改得快"，是"改了之后知道有没有改坏"。

- **`scripts/bench_report.py`**：分段计时 + 峰值 RSS，产出 JSON 入 `.work/bench/`。
  分段而不是只记总时间——各阶段对"数据量翻 10 倍"的响应曲线完全不同，只看总数
  会把一次正确的优化淹没在噪声里。
- **`tests/baseline/` Golden Master**：冻结 51 张榜 × 28 行的全部展示数字、评分卡、
  meta、以及 `window.json`（含天数轴）。判据：名次**零容差**、展示数字逐位相同、
  内部浮点 1e-9。
- **`test_perf_budget`**：性能预算是**棘轮**，只许随优化单向收紧。
- **`src/weather_eval/provenance.py`**：`report.meta.provenance` 注入 `code_sha` /
  `lock_hash` / `bootstrap_seed` / Python 与依赖版本。此前 meta 只有 9 个字段，
  连"这份报告是哪个 commit 算出来的"都不可考——README 承诺的"固定种子可复现"
  因此无法兑现。
- **锁文件 `uv.lock`（38 个包，带哈希）** + `scripts/install.sh`（一条命令装好，
  内含 cyeva 的 `--no-deps` 两步），+ PR 触发的 `.github/workflows/ci.yml`，
  + pytest 标记体系（unit / contract / golden / slow / parity）。

> 两次"假红灯"都值得留档，它们都发生在工具本身而不是被优化的代码：
> （1）基线序列化把浮点 round 到 6 位 → `scorecard[..].r` 差 4.6e-7 超过 1e-9 容差，
> 红了 20 行而报告其实一位没变；**基线不做任何舍入**。
> （2）见 §2 的 cyeva 舍入陷阱。

---

## 2. 分级指标：70% 的 cyeva 耗时，零消费（Phase 1）

`scripts/cyeva_audit.py` 的三个答案（详见 `docs/cyeva-call-audit.md`）：

| 问题 | 方案预估 | 实测 |
|---|---|---|
| 重复率（能否 memo） | 减 20–40% 调用 | 重复率 23.4%，但**可回收仅 0.02 s** |
| 门槛后置浪费 | 有 | 无（门槛本来就在 cyeva 之前） |
| 分级指标是否被消费 | — | **零消费**（页面不渲染、摘要不落盘） |

**与方案不一致的两处决策**：

1. **不实现 memo 缓存**（方案 TASK-03 的一半）。那些"重复"全是空桶——空输入哈希
   到同一个键，函数在构造 cyeva 对象之前就提前返回了，1,092 次里有 303 次、1,540 次
   里有 414 次是空输入，合计 0.06 s。为 0.02 s 引入一份常驻缓存 + 一套失效规则
   不成立。
2. **分级指标走 numpy 快速路径**（`src/weather_eval/graded.py`）。它占指标耗时的
   70%（26.65 s / 38.06 s），896 次调用展开成 34,496 次 cyeva 子调用。而它的数学
   只有"一次区间比较 + 4 个计数"，按级别可加 → 一次向量化扫描给出全部级别。
   cyeva **保留**为权威实现与对拍参照（`eval.graded_backend: cyeva` 可切回）。

**对拍前置条件已满足**：`scripts/parity_graded.py` 在真实数据上逐格比对
**711 个格子 × (5/6 级) × 7 项 ≈ 4.6 万个数字，零不一致**。

### 复刻时抓到的口径陷阱

`calc_precip_*_indicators` 上的装饰器写作 `@source_round_digit(2)`，而签名是
`source_round_digit(series_num=2, digit_num=1)`——那个 `2` 是**要舍入几个位置参数**，
不是小数位数。真实位数是默认的 **1**。第一版按 2 位实现，对拍立刻红了 200 处
（`acc` 75.83 vs 79.46）。没有对拍脚本，这个错误会以"报告数字悄悄变了"的形式上线。
`tests/test_graded_parity.py::test_graded_rounding_uses_one_decimal` 把它钉住了。

---

## 3. 列式化：把"每格扫一遍记录"变成"全程扫一遍"（Phase 2）

`n_eff` 有 2,044 次调用、每次重扫同一批记录（13.7 s）；充分统计量表再扫一遍
（15.3 s）。新增 `src/weather_eval/pairtable.py`：把配对事实存成 numpy 列，
每个 (模型, 天桶) 格子是一段**下标**而不是一份新 list。

- `_n_eff_*_columnar`：min-lead 去重由 `np.lexsort` 完成（稳定排序 → lead 并列取
  最先出现；时刻按字符串升序；站序按首次出现），之后仍交给**未修改的**
  `n_eff_from_station_series`。
- `stats.build_day_stat_tables_columnar`：11 项充分统计量全部向量化，`np.add.at`
  的累加顺序与记录顺序一致 → 浮点位都不差。
- **`collect()` 的输出契约未变**：dict 列表仍是报告其余部分的输入。列式表是
  **并列的、可重建的视图**，删掉它一切照旧。

**I4 是这里的第一约束**：列式化天然把缺测变成 NaN，而 NaN 会被 `>=` 静默当数值。
故显式保存 `*_ok` 布尔列（且用 `isfinite` 而不是 `is not None`），**判定只看 `_ok`**。

三条不变量锁在 `tests/test_columnar_parity.py`：
n_eff **整数零容差**、充分统计量表 `array_equal` **零容差**、缺测入样判定逐条一致。

### 对拍抓到的真 bug

`_n_eff_daily_temp` 的列式版第一版写成了 `emax*pmax + emin*pmin`——未成对的那个量
在列里是 NaN，而 **`nan * 0.0 == nan` 不是 `0.0`**，于是日温度 n_eff 从 3 掉到 2。
改成"先按成对标记换成 0 再加"。这类错误在数值上表现为"少算一点"，看代码看不出来。

---

## 4. 附带修复：诊断层对冷层无感知

把诊断层改成复用内存快照（TASK-04）时，发现它此前**根本读不到冷层数据**：
`iter_error_samples` 逐个文件读入后取 `snap["issue_iso"]`，而**月度 bundle 的根节点
是容器**（`__bundle__` / `snapshots`），没有 `issue_iso` → 全部 `continue`。

| | 修复前 | 修复后 |
|---|---:|---:|
| 参与诊断的样本数 | 6,888 | **674,366** |
| `mean_rho` | 0.5815 | 0.6013 |
| `k_eff`（"27 家相当于几家独立信源"） | 1.68 | 1.62 |
| 指纹漂移告警源 | 3 个 | 4 个 |

诊断层的招牌数字此前是用 3 天数据算出来的。代价是耗时 +26 s——用 26 秒换一个
此前失真的结论，显然值得。这也是本次总耗时只降 19% 的原因。

---

## 5. 并行化：实测为负收益，故改为自适应

方案 TASK-02 预期 `precip_metrics` 28 s → 9 s。但分级指标向量化之后，cyeva 只剩
约 17 s 可并行，而**并行化的成本随父进程内存增长**——`build_report` 此刻已持有
约 1.9 GB 配对数据，fork 4 个 worker 的写时复制把这份内存反复触碰：

| | 总耗时 | 诊断层 | collect |
|---|---:|---:|---:|
| 并行（4 worker） | 114.9 s | 35.2 s | 10.6 s |
| **串行** | **92.5 s** | 29.6 s | 6.8 s |

并行机制**保留**（`src/weather_eval/parallel.py` + 确定性测试），但默认按
**fork 是否划算**自适应：父进程 RSS > 1.2 GB 时退回串行
（`parallel.FORK_RSS_LIMIT_MB`）。等到事实层不再把全量配对驻留内存（增量 IO），
或数据量涨到 13 个月（cyeva 工作量 ∝ 样本量，届时约 100 s 可并行），
这条闸门会自动放行。可用 `WEATHER_EVAL_METRIC_WORKERS` 强制指定。

---

## 6. 工程底座

- `pyproject.toml` 补 `[project]` + entrypoint `weather-eval` + `src/weather_eval/cli.py`
  （薄壳，避免两个入口漂移）+ ruff/mypy 配置 + pytest 标记。
- `uv.lock`（38 包，含 `override-dependencies = ["pint==0.24.4"]` 解决 cyeva 冲突）。
- `scripts/install.sh`：一条命令装好，含 3.12 的 Python 版本门禁。
- `ruff check src scripts tests` → **0 error**（修了 83 处：未用 import/变量、
  lambda 赋值、`;` 同行、超长行、import 位置）。
- `.github/workflows/ci.yml`：PR 触发的快层 + golden 慢层 + 3.13 就绪度（allow-failure）。
  此前仓库没有 PR 触发的测试，533 个用例只在定时运行里被顺带执行。

---

## 7. 明确没做的（以及为什么）

| 项 | 状态 | 理由 |
|---|---|---|
| memo 缓存（TASK-03 的一半） | **不做** | 实测可回收 0.02 s（§2） |
| 分级指标的"不算" | 改为"快速算" | 两个测试断言 `graded` 结构存在，它是报告数据契约的一部分；改成不算等于删数据。故保留数据、换实现 |
| Parquet 事实层与完整增量失效（TASK-06/09/10 的 IO 部分） | 未做 | 列式化已吃掉这部分的主要收益（20 s）；真正的"只读增量"需要 §4.4 四条失效规则与水位表，是独立的一个 PR |
| SQLite 证据层（TASK-12/13） | 未做 | Phase 3，需双写 + 双通道 verify |
| 数据/代码仓分离（TASK-14） | 未做 | 前置是 `available_months()` 改造 |
| 前端数据外置 / JS 模块化（TASK-20/21/22） | 未做 | Phase 5，独立 PR |
| 语义债 #1/#2/#6（会改口径） | 未做 | 按方案要求，口径变更必须是一次独立的、有 README 记录的决策，绝不能混在性能优化里 |

---

## 8. 复现方式

```bash
./scripts/install.sh                      # 一条命令装好
python -m pytest -m "not golden and not slow" -q    # 快层
python -m pytest -m golden -q             # Golden Master 对拍 + 性能预算
python scripts/bench_report.py --window-file tests/baseline/window.json
python scripts/cyeva_audit.py --window-file tests/baseline/window.json
python scripts/parity_graded.py --window-file tests/baseline/window.json
```
