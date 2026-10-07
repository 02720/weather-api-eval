"""配置加载：stations.yaml。"""
from __future__ import annotations

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
    "min_board_neff": 30,               # 进入总榜排名的有效样本量门槛（n_eff，考虑误差
                                       # 自相关后）；未达标源列"样本积累中"不参与冠军竞争
    "min_board_neff_daily": 20,         # 日榜温度维的入围门槛（2026-09 双轨道新增）。
                                       # 门槛必须跟 n_eff 的**计数单位**走：逐小时温度
                                       # 的 n_eff 数的是"独立小时误差"，日最高/最低的
                                       # n_eff 数的是"独立自然日"——同一批存档后者往往
                                       # 只有前者的几十分之一。套用 30 会把所有源一刀
                                       # 切掉。20 与 min_board_neff_rain 同尺度（都是
                                       # 按天计数，≈5 天 × 4 站）。
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
    # ---- 评分换算斜率（2026-10-05 标定，第一性原理审查 P1-1 的落地）----
    # 一项指标对名次的实际影响力 = 权重 × 换算斜率 × 数据离散度，三者耦合。
    # 旧评分表只声明了权重，斜率（RMSE×5、MBE×10、BIAS−1 线性）无人标定，
    # 2026-09 实测 mae 名义 10% 的实际话语权只有 2.5%、bias 名义 10% 却有 27%——
    # 名次由数据的偶然分布决定，README 的权重承诺没有兑现。
    # 此处的 λ 按"子分桶内跨源 sd = 8 分"在真实存档上反解（影响力 = w×8，
    # 占比自动 ≈ 名义权重），标定脚本 scripts/calibrate_score_slopes.py，
    # 方法论与分轨残差留档 docs/score_slopes.md。**季度重标定；λ 变更视为评分
    # 口径变更，月报注明**（变更前后名次不直接可比）。
    #   2026-10-06 全指标审计后入分项从 9 增到 13（温度 +acc1/+mae，降水
    #   +amt_mae/+amt_bias）；同日第二轮审计再增 2（温度 +mbe_bdisp、降水
    #   +grade_ets，共 15 项），标定与对拍留档 docs/metric_inclusion_v2.md。
    #   新增项同法标定（scripts/calibrate_score_slopes.py，
    #   留档 docs/score_slopes.md §八）。**χ²/RSS/TS/acc/漏报率/POFD/雨量 RMSE
    #   不是"没标定所以不入分"，而是数学上是入分项的函数或样本量的函数**
    #   （恒等式见 scripts/audit_metric_redundancy.py 与 docs/metric_coverage.md）。
    "score_slopes": {
        # 温度 7 项（acc2/acc1/rmse/mae/r/mbe/slope）
        "acc2": 0.563,  # ±2°C 命中率每差 1.78 个百分点扣 1 分（×0.563 直接入分）
        "acc1": 0.572,  # ±1°C 命中率每差 1.75 个百分点扣 1 分（命中轮廓族的第二个点）
        "rmse": 12.344, # 每差 0.081 °C 扣 1 分，≥8.1 °C 记 0 分（100−RMSE×12.34）
        "mae": 16.251,  # 每差 0.062 °C 扣 1 分，≥6.2 °C 记 0 分（误差幅度族的 L1 侧）
        "r": 64.511,  # r×64.5（−1~1 线性入分，负值记 0 分）
        "mbe": 13.621,  # 每差 0.073 °C 扣 1 分，≥7.3 °C 记 0 分（正负偏差对称）
        "slope": 13.191, # 幅度每偏 2 倍（log₂ 刻度）扣 13.2 分，超/欠对称
                         # （旧式 100−|slope−1|×100 不对称且日轨 8.9% 格子归零）
        # 降水 6 项（ets/pod/far/bias + amt_mae/amt_bias）
        "ets": 129.039, # ETS×129（0~1 线性入分，负值记 0 分）
        "pod": 0.599,  # 命中率每差 1.67 个百分点扣 1 分
        "far": 0.748,  # 空报率每高 1.34 个百分点扣 1 分（100−FAR×0.748）
        "bias": 15.785, # 报雨频率每偏 2 倍（log₂ 刻度）扣 15.8 分，超/欠报**对称**
                         # （旧式 100−|BIAS−1|×100：超报 2 倍即 0 分、欠报一半
                         #  却得 50 分，既不对称又把日轨 56.9% 的格子压死在 0）
        "amt_mae": 4.188,  # 相对雨量误差（雨量 MAE ÷ 实况平均雨量）每高 0.24 扣 1 分
                           # （**必须**用相对口径：裸 mm 的跨桶尺度差 20 倍，单一 λ
                           #  标不出来，入分等于让按天桶独占该维话语权）
        "amt_bias": 10.983, # 雨量总量每偏 2 倍（log₂ 刻度）扣 11.0 分，超/欠报对称
        # 2026-10-06 第二轮全指标审计后新增两项（温度 +mbe_bdisp、降水 +grade_ets），
        # 同法标定（两轨池化反解，scripts/audit_full_inclusion.py 同口径），留档
        # docs/metric_inclusion_v2.md §七。分轨残差：grade_ets 小时 4.17/日 10.50、
        # mbe_bdisp 小时 6.16/日 8.86——跨轨比 2.52×/1.44×，与既有项（slope 2.27×）
        # 同量级，受"λ 按轨分别标定仍未做"的同一局限约束（docs/score_slopes.md §七）。
        "grade_ets": 177.504,  # 雨强分辨力（中雨/大雨档 ETS 加权均值）×λ，0~1 线性入分，负值记 0 分
        "mbe_bdisp": 36.098,  # 站间一致性：各站系统偏差的离散度每差 0.028°C 扣 1 分，≥2.77°C 记 0 分
                            # （与 BIAS 同纪律；整桶雨一滴没报 → 总量比 0 → 记 0 分）
    },
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


class Config:
    def __init__(self, data: dict):
        self.raw = data
        self.models: list[str] = list(data.get("models", []))
        self.stations: list[Station] = [Station(s) for s in data.get("stations", [])]
        self.eval: dict[str, Any] = {**DEFAULT_EVAL, **(data.get("eval") or {})}
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
