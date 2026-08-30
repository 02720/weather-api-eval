# 天气预报 API 准确度评估系统

自动抓取多个来源的数值天气预报，与事后到达的实况观测对比，用 [`cyeva`](https://github.com/caiyunapp/cyeva) 评估**气温**与**降水**的准确度，生成可在线浏览的多模型对比排行榜（GitHub Pages 部署后即网站首页 `reports/index.html`）。

目前接入 9 个预报源、24 个模型（ECMWF/GFS/ICON/CMA/GRAPES 等国际主流模式，以及彩云、和风、中科天机、伏羲、风乌、中科星图、AccuWeather 等国内可用的商业与 AI 预报），覆盖 4 个广西/海南气象站。

## 核心设计：存档先行

预报检验的第一性原理：**预报必须在实况出现之前存档固定**，事后才与观测配对比对，否则任何"评估"都可能被事后取数污染。因此系统的工作方式是：

1. 每次定时运行，把当时能拿到的**起报快照**（起报时刻 + 逐小时序列）原样封存进 `data/`，永不修改；
2. 观测到达后，按时间戳把封存的预报与观测逐点配对；
3. 每个样本按其**实际时效**（lead = 有效时刻 − 起报时刻）归入对应的「提前 N 天」天桶，这是连续数值预报检验的标准做法；
4. 严格模式：不回填历史起报，数据从首次运行起逐日积累，样本随时间自然变厚。

## 预报源

| `--source` | 来源 | 模型（config 中的名字） | 凭据（环境变量） |
|---|---|---|---|
| `open_meteo`（默认） | [Open-Meteo](https://open-meteo.com) | `ecmwf_ifs`、`ncep_gfs_global`、`dwd_icon_global`、`best_match`、`cma_grapes_global`、`cmc_gem_gdps`、`jma_gsm`、`ukmo_global_deterministic_10km`、`ecmwf_ifs025`、`ecmwf_aifs025_single`、`ncep_aigfs025`、`ncep_hgefs025_ensemble_mean` | 无需 |
| `caiyun` | 彩云天气 v2.6 | `caiyun_v2_6` | `CAIYUN_TOKEN` |
| `qweather` | 和风天气 weather/v1 | `qweather_v1` | `QWEATHER_API_KEY`（建议同时设 `QWEATHER_API_HOST`） |
| `tianji` | 中科天机（网页接口） | `tj_km_fusion`、`tj_t2_early`、`tj_t2`、`tj_t1`、`tj_t1h_ai` | 无需 |
| `fuxi` | 伏羲中期 FuXi-C88（网页接口） | `fuxi_c88` | 无需 |
| `fuxi_data` | 伏羲确定性 FuXi-Det（数据服务 API） | `fuxi_det` | `FUXI_DATA_TOKEN` |
| `fengwu` | 风乌 FengWu-GHR-9km（网页接口） | `fengwu_ghr_9km` | 可选 `FENGWU_API_KEY`（解锁逐小时 360h） |
| `geovis` | 中科星图 GeoVis（官方 API） | `geovis_v1` | `GEVIS_TOKEN` |
| `accuweather` | AccuWeather（官方 API） | `accuweather_v1` | `ACCUWEATHER_API_KEY` |

观测源为[环境气象数据服务平台](http://eia-data.com)（eia-data.com），抓取各气象站"气象站基本信息"页的近 24 小时逐小时实况（气温、降水）。Open-Meteo 一次请求即返回全部模型；其余源为独立抓取，接口契约细节见 `src/weather_eval/forecast/` 各模块顶部 docstring。

## 快速开始

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pip install --no-deps cyeva==0.2.3   # 必须单独装，见下方"依赖说明"

export PYTHONPATH="$PWD/src"         # src 布局且未打包，需手动加导入路径

python -m weather_eval fetch-obs                                # 抓观测
python -m weather_eval fetch-forecast                           # 抓 Open-Meteo 快照
python -m weather_eval fetch-forecast --source tianji           # 其他源同理，凭据先设环境变量
python -m weather_eval report                                   # 生成 reports/index.html
```

抓取是幂等的（按 站点×模型×起报时刻 去重），重复运行不产生冗余数据；单次失败不丢数据，下次运行会自动补回近 24h 的观测窗口。

### 命令一览

| 命令 | 作用 |
|------|------|
| `fetch-obs` | 抓取各站近 24h 观测并归档 |
| `fetch-forecast [--source X]` | 抓取预报起报快照并归档（源见上表，默认 Open-Meteo） |
| `report` | 用"本月 1 号至今"的累计数据覆盖更新主报告 `reports/index.html` |
| `monthly [--month YYYY-MM]` | 把某月冻结为月度归档 `reports/monthly/YYYY-MM.html`（默认上一自然月），随后重建主报告 |
| `all` | `fetch-obs` + `fetch-forecast` + `report`，GitHub Action 的主步骤 |

### 依赖说明（重要）

- Python 须为 **3.12**（cyeva 0.2.3 不支持 3.13）。
- cyeva 的元数据把 `pint` 误钉为 0.24.3，而 0.24.3 在 Python 3.12+ 上无法导入，故必须用 `--no-deps` 单独安装 cyeva，由 `requirements.txt` 提供可用的 `pint==0.24.4` 及其真实依赖（pandas/scipy）。升级 pint 版本前先确认 cyeva 仍可导入。

## 自动运行（GitHub Actions）

1. 推送仓库（需包含 `data/` 与 `reports/`），在 **Settings → Pages** 把部署源选为 **GitHub Actions**。
2. 工作流 `.github/workflows/eval.yml` 每天**北京时 06:00 / 13:00 / 20:00** 运行：抓观测和各源预报 → 覆盖更新主报告 → 提交数据 → 部署 Pages。每月 1 号 06:00 那次运行额外把上一自然月冻结为月度归档（也可 `workflow_dispatch` 手动指定月份）。
3. 各商业源的凭据配置为同名 Secret（`CAIYUN_TOKEN` 等）；**未配置的源自动跳过**，已配置但抓取失败只标注该步骤、不阻断主流程。

## 评估方法

- **配对**：预报与观测按北京时整点时间戳精确配对。观测 `rain@t` 覆盖 (t−1h, t]，各预报源的逐小时降水口径（累计窗口、单位、起报偏移）在各自 provider 内统一转换为"前 1 小时累计"后再入库，偏移量可在 `config/stations.yaml` 的 `precip_offset_hours` 调整。
- **样本与时效**：每个（起报, 有效时刻）组合是一条独立样本，按实际时效归入 1–16 的「提前 N 天」天桶；短时效样本天然更多，时效曲线正是要反映"越临近越准"。另做按天评估（北京时自然日的日最高/最低气温、日降水量）与 1–72h 逐小时曲线。
- **样本下限**：所有指标附样本数 n，`n < 5`（`min_sample`）视为"样本不足"，不出结论。
- **指标**：温度 9 项（RMSE/MAE/MBE/±1°C·±2°C 准确率/相关系数/回归斜率/RSS/χ²），降水 11 项（晴雨二分类 8 项：准确率/POD/FAR/POFD/漏报率/TS/ETS/BIAS ＋ 雨量连续量 3 项），另有逐小时雨强 5 档与 24h 累计 6 档的分级指标。全量定义见 `evaluate.py` 模块 docstring。

### 评分体系（排行榜）

综合得分 =（温度分 + 降水分）/ 2，每个子分由互不重复维度的代表性指标加权而成，统一换算到 0~100；某指标缺数据时按剩余权重归一（不因缺项扣分，缺失维度在榜单以 — 明示）。

- **温度分**：±2°C 准确率 25% ＋ RMSE 换算分 25% ＋ 相关系数 r 15% ＋ ±1°C 准确率 10% ＋ MAE 换算分 10% ＋ |MBE| 偏差换算分 10% ＋ 回归斜率换算分 5%。
- **降水分**：TS 30% ＋ ETS 25% ＋ 晴雨准确率 15%（干燥期天然偏高故低权重）＋ POD 15% ＋ 100−FAR 10% ＋ |BIAS−1| 换算分 5%。

权重表在 `evaluate.py` 的 `TEMP_SCORE_PARTS` / `PRECIP_SCORE_PARTS`，是评分、页面"评分构成"表与本文件三处的单一数据源。RSS、χ²、雨量 RMSE/MAE/MBE 等指标计算但不入分（理由见代码注释与页面说明）。

## 报告

- **主报告 `reports/index.html`**：展示"本月至今"的累计评估，每次运行覆盖更新（数据本体在 `data/` 持续积累，报告只是当前数据的一个视图，不堆文件）。页面按"榜单优先"组织：冠军结论 → 表格式排行榜（每个「提前 N 天」一张榜，可切时效、排序、搜索）→ 得分随时效衰减趋势 → 气温/降水对比图 → 实况对比/热力图/分站图 → 进阶区（按天评估、分级降水、全指标明细表）→ 名词词典。
- **月度归档 `reports/monthly/YYYY-MM.html`**：每月 1 号把上月冻结，永不改动；归档后立即重建主报告使归档出现在首页。
- 每月 1 号主报告切换到新月份窗口后，页面短暂回到"样本积累中"属正常现象，上月完整结果在归档里。

## 目录结构

```
config/stations.yaml      站点、模型、评估参数
src/weather_eval/
  timeutil.py             北京时工具
  config.py  storage.py   配置与 JSON 存档（按月合并文件、文件锁、原子写、幂等）
  obs/                    观测源（eia-data 抓取解析）
  forecast/               各预报源快照器（base.py 定义 ForecastProvider 接口，各模块 docstring 载有接口契约细节）
  evaluate.py             配对 + cyeva 指标 + 评分 + 报告数据组装
  report/                 Jinja2 + ECharts 报告渲染
  __main__.py             CLI
data/                     观测档案 / 预报起报快照（git 跟踪，持续积累）
reports/                  index.html 主报告 / monthly/ 归档 / vendor/ ECharts 本地副本（git 跟踪，部署 Pages）
tests/                    pytest（含 cyeva 手算对拍）
```

## 扩展

- **新增站点**：在 `config/stations.yaml` 的 `stations` 下加一项（`id`/`name`/`lat`/`lon`/`obs_url`，`obs_url` 为 eia-data 该站"气象站基本信息"页的 URL）。
- **新增模型**：Open-Meteo 模型直接加入 `models` 列表；中科天机模型需先在 `forecast/tianji.py` 的 `MODEL_SPECS` 登记映射。
- **新增预报源**：实现 `forecast/base.py` 的 `ForecastProvider` 接口（返回统一结构的起报快照，支持共享时间轴的 dict 或各模型独立快照的 list 两种形态），在 CLI 注册一个 `--source` 值即可，评估/报告逻辑无需改动。现有 `caiyun.py`/`qweather.py`/`geovis.py`/`accuweather.py` 可作参考，覆盖了官方 API、网页接口抓取、档位回退等常见形态。

## 已知约束

- **非官方网页接口源**（中科天机、伏羲中期、风乌）：抓取的是其可视化页面背后的接口，游客态可用但无官方契约，页面改版即可能失效（失败只影响该源，不阻断其他源）。
- **彩云天气**：返回点数与 User-Agent 强相关，代码中固定的 UA 勿改（改了会被静默截断至约 48 小时）；逐小时时间戳锚定在请求时刻，已下取整到整点配对。
- **AccuWeather**：其服务条款限制将数据用于"评级、排名、评审"类用途，与本项目公开排行榜的定位存在潜在冲突，是否接入请自行评估。定位是"最近城市吸附"而非格点，样本代表最近城市。
- **eia-data 观测源**：页面结构改版会影响解析（已做双路回退），届时需调整 `obs/eia_data.py`。
- **Open-Meteo 免费档**约 1 万次/天、无需 Key，但批量回溯请控制频率。
