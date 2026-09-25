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
                                       # 扫描见 scripts/calibrate_daily_threshold.py）
    "rain_hourly_threshold_mm": 1.0,   # 降水分（小时榜·降水维）阈值：该小时 ≥1mm 记"在下雨"
                                       # （2026-09-24 标定，扫描见 scripts/calibrate_hourly_threshold.py）
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
    "daily_min_hours": 20,             # 按天评估的日覆盖门槛：观测/预报任一侧当天
                                       # 非缺测小时数低于此值，该天该要素不入样
                                       # （防"缺测折算 0.0"与"部分日累计偏低"伪装成技巧）
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
