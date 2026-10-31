"""配置加载：stations.yaml。"""
from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "stations.yaml"

# 评估默认参数（可被 stations.yaml 的 eval 段覆盖）
DEFAULT_EVAL = {
    "temp_accuracy_limits": [1, 2],   # ±1°C、±2°C 准确率
    "rain_threshold_mm": 0.1,          # 有无降水阈值（国内业务：≥0.1mm 记为有降水）
    "rain_daily_threshold_mm": 1.0,    # 降水分（日榜·降水维）阈值：24h 累计 ≥1mm 记"有效降水日"
                                       # （2026-09-06 标定：逐小时 0.1mm 口径全模式 ETS≤0.054
                                       # 无区分度，日累计 1mm 阈值下 ETS 上限恢复到 0.25，
                                       # 扫描见 scripts/calibrate_daily_threshold.py，
                                       #  曲线留档 docs/calibrate_daily_threshold.md）
    "rain_hourly_threshold_mm": 1.0,   # 降水分（小时榜·降水维）阈值：该小时 ≥1mm 记"在下雨"
                                       # （2026-09-24 标定，扫描见
                                       #  scripts/calibrate_hourly_threshold.py，
                                       #  曲线留档 docs/calibrate_hourly_threshold.md）
                                       # 逐小时 0.1mm 口径下模型报雨频率是实况的 2.7 倍（毛毛雨
                                       # 偏差），ETS 中位数 0.078 且分数实际上在给"谁更少毛毛雨"
                                       # 排序；阈值提到 1mm 后预报/实况基率趋于一致（5.66% vs
                                       # 4.64%，BIAS 中位数 2.69→1.26），ETS 中位数升到 0.111，
                                       # 分数才开始测量真技巧。业务上"1 小时下 0.1mm"≈没下，
                                       # 而 ≥1mm/h 才是读者认定的"在下雨"。
    "hourly_lead_days": 16,            # 逐小时评估最大时效（天），即 lead 1..384h
    "daily_max_offset_days": 16,       # 按天评估最大日偏移（天），即 offset 1..16。
                                       # 超出该范围的日预报不评测（预报时效太长无
                                       # 业务意义），且入库时写路径会把日产品的
                                       # 超范围部分截除（snapshot_meta.truncate_daily_block）
    "daily_min_hours": 20,             # 按天评估的日覆盖门槛：逐小时聚合路径取
                                       # "观测/预报共同有值的小时数"（两侧同集合），
                                       # 低于此值该天该要素不入样；日产品补位路径
                                       # 沿用观测全天覆盖门槛（防"缺测折算 0.0"、
                                       # "部分日累计偏低"与"小时集合错位"伪装成技巧）
    "min_sample": 5,                    # 样本数低于此值视为"样本不足"，不出结论
    "min_board_neff": 30,               # 仅保留给诊断层口径走廊披露；2026-10-31 起
                                       # 分块榜入围改由"7 天 + 有效块"决定（下）
    "min_board_neff_daily": 20,         # 同上（日榜温度维的旧口径走廊）。
                                       # 门槛必须跟 n_eff 的**计数单位**走：逐小时温度
                                       # 的 n_eff 数的是"独立小时误差"，日最高/最低的
                                       # n_eff 数的是"独立自然日"——同一批存档后者往往
                                       # 只有前者的几十分之一。套用 30 会把所有源一刀
                                       # 切掉。20 与 min_board_neff_rain 同尺度（都是
                                       # 按天计数，≈5 天 × 4 站）。
    # ---- 降水事件证据门槛（2026-10-23 引入，2026-10-31 改为按 7 天块标定）----
    # n_eff 数的是"雨/无雨判定"序列，干燥小时占绝对多数：一个只覆盖 11 天、只有
    # 14 个 ≥1mm/h 观测事件的源，仍能靠 1,068 个"判定"通过 min_board_neff=30。
    # 门槛量的是**样本量**，不是**降水事件证据量**。这两道门直接数独立事件
    # （站×时刻 / 站×日 去重，同一事件不因多轮抓取重复计）。分块模型下它们
    # 在**单个有效块**内考核，与 7 天新源的证据量同尺度。
    # （min_board_neff* 仍保留在 meta 里供诊断层的口径走廊披露，但不再作为
    #  分块榜的入围门槛——新源永远攒不到累计 30/8。）
    "min_board_rain_events": 8,        # 小时轨：**单个有效块内**观测 ≥1mm/h 的独立
                                       # （站,小时）事件数门槛（2026-10-31 起按 7 天块标定；
                                       # 旧值 30 是累计窗口口径，会把 7 天新源一刀切掉）
    "min_board_rain_days": 3,          # 日轨：**单个有效块内**观测日雨量 ≥1mm 的独立
                                       # （站,日）雨日数门槛（旧值 8 同理）
    # ---- 入设计格子数门槛（2026-10-31 分块模型）----
    # α 的精度取决于**跨了多少个块 × 每块有几个天桶**。分块模型要求一个源至少
    # 在一个块里有 3 个综合分格子，否则行效应无法与块/天桶效应分开。
    "min_board_cells": 3,
    # ---- 7 天日历分块（方案 A，2026-10-31）----
    # 单榜的核心：把时间按周一锚定的 7 天块切开，块作为加法模型的第二个列轴。
    # 门槛跟着"块"走：总样本 ≥7 天 + 至少一个有效块（块内 ≥5 天、小时轨 ≥8 个
    # ≥1mm/h 事件 / 日轨 ≥3 个 ≥1mm 雨日、每条轨道 ≥3 个综合分格子）。
    # 达标源得到**正式**分数/名次/CI；不足 7 天的源显示"样本积累中"，不发布分数。
    "board_block_days": 7,
    "board_min_qualified_days": 7,
    "board_block_min_days": 5,
    "board_block_min_rain_events": 8,
    "board_block_min_rain_days": 3,
    "board_block_min_cells": 3,
    # ---- 降水分"最小存活权重"门槛（2026-10-18，score_review_proposal §2 P8 的修法）----
    # 降水维 5 项入分指标里，ets/pod/far/grade_ets 4 项**需要真实降水事件**才
    # 存在（命中/漏报/空报、各雨强档），amt_mae 一项只要 Σ实况雨量>0 就能算。
    # 于是"整桶只有一两场小雨、没有任何达标事件"的格会退化成只剩雨量一项，
    # 而 _weighted_score 的"缺项按剩余权重归一"会把它**抬成全部话语权**
    # ——实测（冻结基线窗口）5 个格因此拿到 100.0 降水分、11 个格产出 ≥95/≤5 的
    # 极端分，且全部进总榜设计，把两个分时效榜的榜首制造了出来。也即必须区分
    # 两种缺失："这项指标对该源不存在"（重归一是对的）vs"该桶没有可评降水事件"
    # （证据缺失，整维判缺测）。口径：**换算后仍有效的子分权重之和**低于名义
    # 总权重的该比例时，降水分判 None，不重归一。0.5 = 少于半数名义权重可评即
    # 弃（与 temp_score 的 _mean2"半截证据不能当整个用"同一原理，MSN 教训）；
    # 0 = 关闭（旧口径，留作对照）。对温度维无涉：其 8 项在样本存在时恒可算。
    "precip_min_surviving_weight": 0.5,
    # ---- 总榜的双向加法拟合（对抗式审查 P0-1/P1-4）----
    "board_cell_weighting": "neff",     # 格子权重口径："neff" = √(格子有效样本量)（默认，
                                       # 让信息多的格子说话）；"equal" = 等权（旧口径，
                                       # 保留以便对照/回归；两者名次差异会并列披露）
    "min_cell_neff": 15,                # 单个（源, 天桶）格子的最低有效样本量：低于门槛的
                                       # 格子不进劈分设计（旧实现靠 min_sample=5 放行，
                                       # 实测 5 条降水样本即可产出 0.0 分并等权进榜）
    "board_min_col_frac": 0.5,          # 长尾桶相对门槛：家数不足"最大桶家数 × 该比例"的
                                       # 天桶不进主设计（0 = 关闭）。实测第 16 桶只有 7 家，
                                       # 其列效应却等量打进所有源的行分（P1-4）
    "board_ridge": 0.0,                 # 列效应的经验贝叶斯收缩强度 λ（0 = 不收缩；
                                       # λ>0 时家数少的桶的"难度"被拉向平均值）
    "board_long_tail_board": True,      # 是否为被主设计剔除的长尾桶单独出一张参考榜
    "macro_weight_range": [0.30, 0.70], # 权重敏感性里"温度占综合分比例"的扰动区间（P1-2）
    # ---- 评分换算斜率（2026-10-18 重标定，第一性原理审查 P1-1 的落地）----
    # 一项指标的**局部敏感度** = 权重 × 换算斜率 × 数据离散度，三者耦合。
    # 旧评分表只声明了权重，斜率（RMSE×5、MBE×10、BIAS−1 线性）无人标定，
    # 2026-09 实测 mae 名义 10% 的局部敏感度只有 2.5%、bias 名义 10% 却有 27%——
    # 名次由数据的偶然分布决定，README 的权重承诺没有兑现。
    # 此处的 λ 按"子分桶内跨源 sd = 8 分"在真实存档上反解（局部敏感度 = 8w，
    # 与名义权重成正比），标定脚本 scripts/calibrate_score_slopes.py，
    # 方法论与分轨残差留档 docs/score_slopes.md。**季度重标定；λ 变更视为评分
    # 口径变更，月报注明**（变更前后名次不直接可比）。
    #   ⚠️ 局部敏感度 ≠ 方差份额：加权和的方差贡献 ∝ w²·σ²，所以即使 σ 全等于 8，
    #   方差份额仍 ∝ w²（大权重项超配、小权重项欠配）。报告页
    #   meta.diagnostics.slope_influence.nominal_vs_effective 两种口径逐项披露；
    #   本文件不再声称"实际影响力 = 名义权重"（2026-10-31 复核 P0-1）。
    #   2026-10-06 全指标审计后入分项从 9 增到 13（温度 +acc1/+mae，降水
    #   +amt_mae/+amt_bias）；同日第二轮审计再增 2（温度 +mbe_bdisp、降水
    #   +grade_ets，共 15 项），标定与对拍留档 docs/metric_inclusion_v2.md。
    #   2026-10-18 第三轮：λ 契约漂移（复核实测 12/15 项桶内 sd 低于 8，最低
    #   amt_mae 4.91，承诺 8）
    #   修复——全 15 项在**冻结基线窗口**（2026-08-01~10-14，29 源）重标，
    #   池化 sd 回到 8.00；同期落地上端饱和门限（标定脚本 §1.5）与展示层
    #   留白规则（见下方 score_index_gain）。留档 docs/score_slopes.md §九。
    #   2026-10-31：amt_bias 已于 10-25 退役，BIAS 退出评分表 → 共 13 项；
    #   权重重构为温度 4 族 / 降水 3 族（族权重是价值判断，λ 不变）。
    #   **χ²/RSS/TS/acc/漏报率/POFD/BIAS/雨量 RMSE
    #   不是"没标定所以不入分"，而是数学上是入分项的函数或样本量的函数**
    #   （恒等式见 scripts/audit_metric_redundancy.py 与 docs/metric_coverage.md）。
    "score_slopes": {
        # 温度 8 项（4 族：误差/阈值/偏差/形态）
        "acc2": 0.650,  # ±2°C 命中率每差 1.54 个百分点扣 1 分（×0.650 直接入分）
        "acc1": 0.630,  # ±1°C 命中率每差 1.59 个百分点扣 1 分（命中轮廓族的第二个点）
        "rmse": 14.661, # 每差 0.068 °C 扣 1 分，≥6.8 °C 记 0 分（100−RMSE×14.66）
        "mae": 19.260,  # 每差 0.052 °C 扣 1 分，≥5.2 °C 记 0 分（误差幅度族的 L1 侧）
        "r": 69.400,  # r×69.4（−1~1 线性入分，负值记 0 分）
        "mbe": 14.697,  # 每差 0.068 °C 扣 1 分，≥6.8 °C 记 0 分（正负偏差对称）
        "slope": 17.746, # 幅度每偏 2 倍（log₂ 刻度）扣 17.7 分，超/欠对称
                         # （旧式 100−|slope−1|×100 不对称且日轨 8.9% 格子归零）
        # 降水 5 项（ets/pod/far/amt_mae/grade_ets；BIAS 已退出评分表）
        "ets": 120.368, # ETS×120（0~1 线性入分，负值记 0 分）
        "pod": 0.633,  # 命中率每差 1.58 个百分点扣 1 分
        "far": 0.749,  # 空报率每高 1.34 个百分点扣 1 分（100−FAR×0.749）
        # BIAS 的 λ（旧值 18.709）随 2026-10-31 去冗余一并退役：BIAS = POD/(1−FAR)
        # 是列联表的确定性函数，且"多一个空报反而加分"的单调性反例无法在对称
        # log 口径下消除；明细列照旧发布，不再占权重、不再需要换算斜率。
        "amt_mae": 37.709,  # 相对雨量误差（雨量 MAE ÷ 实况平均雨量）每高 0.027 扣 1 分
                           # （**必须**用相对口径：裸 mm 的跨桶尺度差 20 倍，单一 λ
                           #  标不出来，入分等于让按天桶独占该维话语权）
        # amt_bias 的 λ（旧值 11.462）随 2026-10-25 去冗余一并退役：它与 amt_mae
        # 桶内原始相关 0.995/1.000，已并入雨量精度族且不再入分；明细列照旧发布。
        # 2026-10-06 第二轮全指标审计后新增两项（温度 +mbe_bdisp、降水 +grade_ets），
        # 同法标定（两轨池化反解，scripts/audit_full_inclusion.py 同口径），留档
        # docs/metric_inclusion_v2.md §七。2026-10-18 与全部 15 项一起重标。
        # 分轨残差：grade_ets 小时 4.4/日 10.4、mbe_bdisp 小时 6.7/日 7.2——
        # 与既有项（slope 小时 6.1/日 8.6）同量级，受"λ 按轨分别标定仍未做"的
        # 同一局限约束（docs/score_slopes.md §七）。
        "grade_ets": 176.001,  # 雨强分辨力（中雨/大雨档 ETS 加权均值）×λ，0~1 线性入分，负值记 0 分
        "mbe_bdisp": 35.286,  # 站间一致性：各站系统偏差的离散度每差 0.028°C 扣 1 分，≥2.8°C 记 0 分
                            # （与 BIAS 同纪律；整桶雨一滴没报 → 总量比 0 → 记 0 分）
    },
    # ---- 展示层"技能指数"缩放（2026-10-31，第一性原理复核 P0-3）----
    # **只作用于展示，不参与评分/名次/CI/显著性**。维度分与综合分在评分管道里
    # 一律保持未缩放原始分；读者看到的 0~100 分数由 evaluate.display_* 在装配
    # 榜面/趋势图时施加一次仿射 g(s)=center+gain·(s−center)（见 stats.index_gain）。
    # 为什么需要展示缩放：13 个子分各自标定到"桶内跨源 sd = 8 分"，但维度分是
    # 权重合计为 1 的加权平均，互不完全相关的子分一平均，sd 就收缩到 ≈2.8——
    # 整榜被压进 ~10 分带，"区分度不足"。gain 由留白规则反解：取标定窗口全部
    # 维度分的 P99.5 映到软上限 98，gain=(98−50)/(P99.5−50)。
    # **名次用的是未缩放原始分**：缩放不再先于缺项归一/截断/维度合成/双向对齐
    # 发生，截断也就不再可能改变名次或显著性（截断只影响屏幕上的数字）。
    # 固定冻结、不按当期 μ/σ 自适应——若某期各家真的一样，分数就一样，绝不伪造
    # 区分度。**变更视为展示口径变更，分数前后不可直接对比、须重冻基线。**
    "score_index_gain": 1.744,  # 维度分展示增益 = (98−50)/(77.53−50)（新族权重窗口反解）
    "score_index_center": 50.0, # 展示中心（0~100 量表的中点）
    # 综合分视图的**总**展示增益：2026-10-31 起把原来的"维度 gain × 综合分 gain"
    # 两层合并成这一层。综合分的 P99.5（66.76）与维度分（77.53）不同，所以单独
    # 反解；只作用于 display_composite_score / display_ci，不改排名。
    "score_composite_gain": 2.863,  # = (98−50)/(66.76−50)，综合分 P99.5=66.76
    # ---- 小时轨与日轨的宏观权重（2026-10-25，复核 P1）----
    # 总榜把两条轨道的列拼进同一个加法模型。旧实现两轨等权（1:1）——但日轨的
    # 日最高/最低、日累计降水对绝大多数源是**从同一份逐小时预报重新分箱**得到的
    # （原生日产品只补了约 7.6% 的格子），两轨行分 Spearman 0.86：名义"两路独立
    # 证据各半"，实际把"日极值"这一类信息的话语权凭空翻了倍。降为 1/3（时:日=2:1）
    # 后名次扰动最小（与现行 Spearman 0.990），且日轨仍保留其真正独立的部分
    # （日最高/最低是小时轨根本不交付的量；日累计降水则与小时降水部分同源）。
    # 仅作用于跨轨合成（总榜 / 综合天榜 / 趋势 all），单轨榜不受影响。
    "daily_track_weight": 0.5,
    "require_complete_snapshots": True, # 是否排除快照契约标了 complete=false 的残缺快照
                                       # （旧存档无该字段，按完整处理——纯增量，不改旧结论）
    "bootstrap_runs": 500,              # 按天分块 bootstrap 重采样次数（置信区间/冠军频率）
    "sensitivity_runs": 500,            # 权重敏感性扰动次数（权重 ±40% 均匀扰动）
    "daily_source_fallback": True,      # 逐小时覆盖不足时，允许用快照自带的逐日
                                        # 预报（daily_time/daily 块）为按天评估补位。
                                        # 只补按天轨道，绝不反推逐小时；关掉即回到
                                        # 纯逐小时聚合的旧口径（用于对照/回归）
    # ---- 观测多源编排（2026-09 新增：eia-data 单点依赖的兜底）----
    "obs_stale_hours": 3.0,             # 观测源"停摆"阈值（小时）：某源最新观测滞后超过
                                        # 该值即判为不可直接采用（记录仍参与合并），
                                        # 由下一优先级的源补位
    "obs_min_hours": 6,                 # 单个观测源一轮至少要有多少小时才算"窗口未截断"；
                                        # 低于该值即降级（明显截断的页面不能当完整窗口用）
    # ---- 数据体积治理（2026-09 新增：让仓库能持续自动化运行）----
    "compact_grace_days": 2,            # 月度冻结宽限期（天）：自然月结束满该天数后，
                                        # 该月的逐份快照才合并为月度 bundle（冻结后永不重写）
    "compact_retain_months": 13,        # 月度 bundle 的保留月数：更早的 bundle 出仓，
                                        # 其结论已由月度报告与该月指标摘要固化
    # ---- 实时总榜的对账窗口（2026-10 新增：总榜跨月累计）----
    "live_window_start": None,          # 实时总榜的窗口起点（"YYYY-MM" 或 "YYYY-MM-DD"）。
                                        # None = 自动：取已归档数据的最早月份（见
                                        # storage.available_months）。总榜回答"到现在为
                                        # 止谁最准"，理应用保留期内全部样本回答——历史月
                                        # 与当月同榜，月初不清零。设为固定日期可钉死起点
                                        # （例如强制"只看 2026-09 以后"做对照）；
                                        # 保留期出仓（compact_retain_months）是自动窗口
                                        # 的天然上界——更早的原始快照已删除，重算无米下锅。
    # 源健康度看板的"陈旧"阈值（小时）：超过该时长未成功抓取即在 reports/health.html
    # 标陈旧并由 CLI 告警。此前只在代码里以 `cfg.eval.get("source_stale_hours", 30)`
    # 读取、DEFAULT_EVAL 里**没有**这一项——"这个阈值可以配、默认是多少"只存在于
    # 调用点，配置者无从得知（第五轮 P3-c）。登记在此即为它的唯一声明处。
    "source_stale_hours": 30,
}



class Station:
    def __init__(self, data: dict):
        self.id: str = data["id"]
        self.name: str = data.get("name", data["id"])
        self.lat: float = float(data["lat"])
        self.lon: float = float(data["lon"])
        self.obs_url: str = data.get("obs_url", "")
        # CMA 公众气象服务网（weather.cma.cn）以 WMO 站号寻址；未配置时该源的
        # 抓取会响亮失败（见 forecast/cma_public.py），绝不猜站号
        self.cma_id: str | None = str(data["cma_id"]) if data.get("cma_id") else None
        # 最美天气（h5-api.zuimeitianqi.com）以行政区 cityCode 寻址；未配置时该源
        # 同样响亮失败（见 forecast/zuimeitianqi.py），绝不在线猜码——geo 接口
        # 吸附的行政区可随兴趣点库漂移，寻址语义必须可复核
        self.zmtq_city_id: str | None = (
            str(data["zmtq_city_id"]) if data.get("zmtq_city_id") else None)


def _station(s: Any) -> Station:
    """构造 Station，把缺字段的 KeyError 换成指明 stations.yaml 条目的错误。

    原实现直接 `Station(s)`，一条缺 `lat`/`lon` 的站点配置会以裸 `KeyError: 'lat'`
    崩在构造里——traceback 里既没有文件名也没有是哪一条站点，改 YAML 的人只能
    逐条核对。这里补上上下文（第五轮 P3-a）。
    """
    if not isinstance(s, dict):
        raise ValueError(
            f"stations.yaml 的 stations 条目必须是映射，得到 {type(s).__name__}: {s!r}")
    try:
        return Station(s)
    except KeyError as e:
        raise ValueError(
            f"stations.yaml 的站点条目缺字段 {e}：{s!r}"
            "（id / lat / lon 为必填）") from e


class Config:
    def __init__(self, data: dict):
        # 空/非映射的 YAML（`yaml.safe_load` 对空文件返回 None）此前一路走到
        # `data.get` 才以 AttributeError 崩掉，报错里既没有文件路径也没有"配置
        # 为空"这个事实——改配置的人对着 traceback 猜不出自己写错了什么。
        if not isinstance(data, dict):
            raise ValueError(
                f"stations.yaml 解析结果不是映射（得到 {type(data).__name__}）："
                "文件为空或顶层不是 key: value 结构")
        self.raw = data
        self.models: list[str] = list(data.get("models", []))
        self.stations: list[Station] = [_station(s) for s in data.get("stations", [])]
        # ⚠️ **深拷贝** DEFAULT_EVAL 再合并（第五轮 P1-3）：`{**DEFAULT_EVAL, **user}`
        # 是浅合并，嵌套容器（`score_slopes` 字典、`temp_accuracy_limits` /
        # `macro_weight_range` 列表）会与模块级常量**共享同一个对象**。任何就地
        # 改动（apply_score_slopes 的 `{**DEFAULT_SCORE_SLOPES, **(slopes or {})}` 是
        # 新建、但别处对 `cfg.eval["score_slopes"][k] = v` 的写法不是）会永久污染
        # DEFAULT_EVAL，让同进程后续每一次 load_config() 都读到被改过的默认值——
        # 而 lru_cache(maxsize=1) 让这类污染在一次长跑里完全静默。
        self.eval: dict[str, Any] = {**copy.deepcopy(DEFAULT_EVAL),
                                     **(data.get("eval") or {})}
        # 观测源优先级（高 → 低）：主源在前，备用源在后。缺省与
        # obs/chain.py 的 OBS_SOURCE_PRIORITY 一致；在此可被配置覆盖，
        # 但**顺序语义**由编排层实现，配置只负责"登记哪些源、谁先谁后"。
        self.obs_sources: list[str] = list(
            data.get("obs_sources") or ["eia_data", "cma_data"])

    @property
    def station_ids(self) -> list[str]:
        return [s.id for s in self.stations]


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return Config(yaml.safe_load(f))
