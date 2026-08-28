# 天气预报 API 准确度评估系统

自动抓取多个来源的数值天气预报，与实景观测对比，用 [`cyeva`](https://github.com/caiyunapp/cyeva) 评估**温度**与**降水**两个指标的准确度，并生成可在线浏览的多模型对比报告。

> 设计原则：预报必须在实况出现之前**存档固定**，事后再与观测配对比对，才是有效的检验。系统每次定时运行都会抓取并永久封存当时发布的起报快照，后续实况到达后再做验证。

## 支持的来源

- **预报（Open-Meteo）**：`ecmwf_ifs`、`ncep_gfs_global`、`dwd_icon_global`、`best_match`、`cma_grapes_global`、`cmc_gem_gdps`、`jma_gsm`、`ukmo_global_deterministic_10km`、`ecmwf_ifs025`、`ecmwf_aifs025_single`、`ncep_aigfs025`、`ncep_hgefs025_ensemble_mean`（一次请求返回全部模型，后续可在 `config/stations.yaml` 继续增删）。
- **预报（彩云天气 Caiyun v2.6）**：模型名 `caiyun_v2_6`，Token 认证（环境变量 `CAIYUN_TOKEN`）。`fetch-forecast --source caiyun` 单独抓取，评估/报告逻辑与 Open-Meteo 完全一致。
- **预报（和风天气 QWeather weather/v1）**：模型名 `qweather_v1`，API Key 认证（环境变量 `QWEATHER_API_KEY`，专属 API Host 见 `QWEATHER_API_HOST`）。`fetch-forecast --source qweather` 单独抓取；请求未来最多 240 小时逐小时（免费档可能被限制为 24 小时，日志会明示），评估/报告逻辑与其他源完全一致。
- **预报（中科天机 TianJi，网页接口抓取）**：模型名 `tj_km_fusion`（公里级融合）、`tj_t2_early`（天机2/DA，即 T2-Early）、`tj_t2`（天机2/ND，即 T2）、`tj_t1`（天机1/ND，即 T1）、`tj_t1h_ai`（T1H-AI，即 T1-AI）。数据来自 `www.tjweather.com/vis/` 可视化页面背后的单点查询接口（游客态可用、无需凭据）；`fetch-forecast --source tianji` 单独抓取。起报为北京时每天 08/20 时两轮，t2/融合系 240h、t1 系 360h 逐小时。
- **预报（伏羲中期 FuXi-C88，网页接口抓取）**：模型名 `fuxi_c88`。数据来自 `fuxi-ai.cn/visual/weather` 可视化页面背后的自有网关（游客态可用、无需凭据）；`fetch-forecast --source fuxi` 单独抓取。逐小时 360 点（15 天），每天 00/12 UTC（北京时 08/20 时）两轮，发布滞后约 1 天属正常。
- **预报（伏羲确定性 FuXi-Det，数据服务 API）**：模型名 `fuxi_det`。数据来自 `fuxi-ai.cn/fuxi-data` 页面的数据服务网关（**需登录后于该页获取查询 Token**，经环境变量 `FUXI_DATA_TOKEN` 注入）；`fetch-forecast --source fuxi_data` 单独抓取。0.1° 分辨率、每天 00/06/12/18 UTC 四轮。
- **预报（风乌 FengWu-GHR-9km，网页接口抓取）**：模型名 `fengwu_ghr_9km`。数据来自 `fengwuai.com/simple-query` 页面的公开查询 API；`fetch-forecast --source fengwu` 单独抓取。游客态 3 小时步长、起报后 166h；**填 `FENGWU_API_KEY`（经 `Authorization: Bearer` 头传递）可解锁逐小时 360h 完整时效**。原生 6 小时累计降水的展开口径见「已知约束」。
- **预报（中科星图 GeoVis，官方 API）**：模型名 `geovis_v1`。《全国城市逐小时预报》产品（专业版 120h，自动按 专业→进阶→基础 档位回退），**需 Token**（datacloud.geovisearth.com 注册 + 开发者认证，环境变量 `GEVIS_TOKEN`）；`fetch-forecast --source geovis` 单独抓取。
- **观测（环境气象数据服务平台 eia-data.com）**：各气象站"气象站基本信息"页，服务端直出近 24 小时逐小时实况（气温、降水、气压、湿度、风）。
- **评估框架**：`cyeva 0.2.3`（温度 RMSE/MAE/MBE/±1°C·±2°C 准确率；降水 0.1mm 晴雨二分类准确率/POD/空报率 FAR/漏报率/TS/BIAS，及分级降水 TS）。

## 评估口径（与报告顶部说明一致）

- **逐小时**：按起报后时效分 1–16 个「天桶」；温度 RMSE/MAE/MBE/±1°C/±2°C 准确率；降水 0.1mm 晴雨二分类指标。另含 1–72h 逐小时 RMSE 曲线。
- **按天**：北京时自然日（00:00–24:00）聚合日最高/最低气温、日降水量；按"有效日 − 起报日"的日偏移 1–16 天分组，评估日最高/最低气温 ±2°C 准确率与日降水 TS。
- 所有指标附样本数 n；**n < 5 视为"样本不足"，不出结论**。
- 降水口径对齐假设：观测 `rain@t`（t−1h→t 累计）对应 Open-Meteo `precipitation@t`（前 1 小时累计）、中科天机 `pratesfc@t`（逐小时降水率 mm/h，作为前 1 小时累计的近似），偏移量在配置中可改。

## 环境要求

- **Python 3.12.6**（cyeva 0.2.3 支持 3.10–3.12；`pint` 必须用 **0.24.4**，因为 cyeva 误钉的 0.24.3 在 3.12+ 上无法导入，见下方安装说明）。
- 网络访问：Open-Meteo（HTTPS）、eia-data.com（HTTP）、彩云天气 API `api.caiyunapp.com`（HTTPS）、和风天气 API `<你的专属 Host>.qweatherapi.com`（HTTPS）、中科天机 `www.tjweather.com`（HTTPS）。Open-Meteo 免费档约 1 万次/天，无需 key；**彩云需 Token**，从环境变量 `CAIYUN_TOKEN` 读取；**和风需 API Key**，从环境变量 `QWEATHER_API_KEY` 读取；**中科天机无需凭据**（抓取其网页可视化的单点查询接口，属非官方契约，若页面改版需相应调整）。

## 快速开始（本地）

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# cyeva 用 --no-deps 单独安装（放最后，绕过其错误的 pint==0.24.3 约束）；先装 requirements 提供兼容的 pint==0.24.4
pip install -r requirements.txt -r requirements-dev.txt
pip install --no-deps cyeva==0.2.3

# 仓库是 src 布局且未做 pip 打包，需把 src 加入导入路径（与 CI 一致）
export PYTHONPATH="$PWD/src"

# 抓取 4 站近 24h 观测
python -m weather_eval fetch-obs
# 抓取 12 个 Open-Meteo 模型起报快照（未来 16 天逐小时）
python -m weather_eval fetch-forecast
# 彩云/和风/中科天机/伏羲/风乌/中科星图为独立源，按上一节命令单独抓取
# 用本月至今的累计数据更新主报告（每次覆盖，不堆文件）
python -m weather_eval report
# 浏览：打开 reports/index.html（GitHub Pages 部署后即网站首页）

# —— 彩云天气（v2.6）接入 ——
# 1) 设置 Token（仅从环境变量读取，**切勿提交到仓库**）
#    彩云 Token 请通过环境变量 / 密钥管理（如 GitHub Actions Secret）注入；
#    测试 Token 由维护者单独提供，不要硬编码或写入本文件。
export CAIYUN_TOKEN="<在此填入你的彩云 Token>"
# 2) 抓取彩云起报快照（独立于 Open-Meteo；评估/报告逻辑无需改动）
python -m weather_eval fetch-forecast --source caiyun
# 3) 生成报告时 caiyun_v2_6 已纳入对比（config 的 models 已含该模型）
python -m weather_eval report

# —— 和风天气（weather/v1）接入 ——
# 1) 设置 API Key 与专属 API Host（控制台「设置」中查看；旧公共域名 devapi/
#    api.qweather.com 自 2026 年起逐步停止服务）。Host 未设置时会退回 devapi 并告警。
export QWEATHER_API_KEY="<在此填入你的和风 API Key>"
export QWEATHER_API_HOST="abcxyz.qweatherapi.com"
# 2) 抓取和风起报快照（独立于 Open-Meteo；默认请求未来 240 小时逐小时）
python -m weather_eval fetch-forecast --source qweather
# 3) 生成报告时 qweather_v1 已纳入对比（config 的 models 已含该模型）
python -m weather_eval report

# —— 中科天机（网页接口抓取，无需凭据）——
# 抓取 5 个模型起报快照（北京时 08/20 时两轮起报，t2/融合系 240h、t1 系 360h）
python -m weather_eval fetch-forecast --source tianji
# 生成报告时 tj_* 模型已纳入对比（config 的 models 已含）
python -m weather_eval report

# —— 伏羲中期 FuXi-C88（网页接口抓取，无需凭据）——
python -m weather_eval fetch-forecast --source fuxi

# —— 风乌 FengWu-GHR-9km（游客态 7 天/3h 步长；填 Key 延长时效）——
export FENGWU_API_KEY="<可选：风乌开放平台 Key，解锁逐小时 360h>"
python -m weather_eval fetch-forecast --source fengwu

# —— 伏羲确定性 FuXi-Det（需 fuxi-data 页面登录后获取的查询 Token）——
export FUXI_DATA_TOKEN="<登录 fuxi-ai.cn/fuxi-data 后页面获取>"
python -m weather_eval fetch-forecast --source fuxi_data

# —— 中科星图（需注册 + 开发者认证获取 Token）——
export GEVIS_TOKEN="<datacloud.geovisearth.com 控制台获取>"
python -m weather_eval fetch-forecast --source geovis
```

常用命令：

| 命令 | 作用 |
|------|------|
| `fetch-obs` | 抓取观测并归档（按时间戳去重） |
| `fetch-forecast` | 抓取 Open-Meteo 起报快照并归档（幂等） |
| `fetch-forecast --source caiyun` | 抓取彩云天气 v2.6 起报快照（需 `CAIYUN_TOKEN`） |
| `fetch-forecast --source qweather` | 抓取和风天气起报快照（需 `QWEATHER_API_KEY`，建议同时设 `QWEATHER_API_HOST`） |
| `fetch-forecast --source tianji` | 抓取中科天机起报快照（网页接口抓取，无需凭据） |
| `fetch-forecast --source fuxi` | 抓取伏羲中期 FuXi-C88 起报快照（fuxi-ai.cn 可视化接口，无需凭据） |
| `fetch-forecast --source fuxi_data` | 抓取伏羲确定性 FuXi-Det 起报快照（需 `FUXI_DATA_TOKEN`） |
| `fetch-forecast --source fengwu` | 抓取风乌 GHR-9km 起报快照（游客态 7 天；`FENGWU_API_KEY` 可选延长） |
| `fetch-forecast --source geovis` | 抓取中科星图逐小时预报起报快照（需 `GEVIS_TOKEN`） |
| `report` | 用"本月 1 号至今"的累计数据更新主报告 `reports/index.html`（覆盖写，不堆文件） |
| `monthly [--month YYYY-MM]` | 把某月冻结为月度归档 `reports/monthly/YYYY-MM.html`（默认上一自然月） |
| `all` | `fetch-obs` + `fetch-forecast` + `report`（GitHub Action 调用；彩云/和风/中科天机/伏羲/风乌/中科星图为独立可选步骤） |

## GitHub Actions 自动运行

1. 在 GitHub 新建仓库，将本项目推送（需包含 `data/` 与 `reports/`，已用 `.gitignore` 排除 `.venv` 等）。
2. 仓库 **Settings → Pages → Build and deployment → Source 选 "GitHub Actions"**。
3. 工作流 `.github/workflows/eval.yml` 已配置：
   - **定时**：北京时每天 **6:00 / 13:00 / 20:00**（cron `0 5,12,22 * * *` UTC）。
   - 每次运行抓取观测+预报，并用本月累计数据**覆盖更新主报告** `reports/index.html`（Pages 首页），提交数据后部署。
   - **每月 1 号北京时**自动把上一自然月冻结为月度归档 `reports/monthly/YYYY-MM.html`（也可手动 `workflow_dispatch` 指定 `month`）。
4. 报告在线地址：`https://<用户名>.github.io/<仓库名>/`（Pages 自动发布 `reports/` 目录，首页即最新主报告）。

> 每次运行仅抓取近 24h 观测，因此单次失败不会丢数据（下次运行自动回补 24h 窗口）。

## 报告体系（怎么看报告）

- **主报告 `reports/index.html`**：网站首页，展示"本月至今"的**累积**评估结果，每次 Action 运行自动覆盖更新。页面按"结论先行"组织：顶部一句话冠军结论 → 30 秒新手引导 → 预报源排行榜 → 气温/降水/实况对比/热力图/分站图表（每张图配"💡 怎么看"导读）→ 进阶数据（默认折叠）→ 名词小词典。
- **月度归档 `reports/monthly/YYYY-MM.html`**：每月 1 号把上月冻结存档，永不改动；`monthly` 命令在归档后会立即重建一次主报告，让新归档在当次部署就出现在首页页脚的归档列表里。
- 数据本体在 `data/`（观测/起报快照）里持续积累，报告只是"当前累计数据的一个视图"——所以不存在"每次运行一份报告"的文件堆叠。
- 注意：每月 1 号主报告会切换为新月份的评估窗口，页面短暂回到"样本积累中"状态属正常现象（上月的完整结果已冻结在归档里）。

## 关于"首份完整月报"

系统采用**严格模式**（不在事前回填历史起报），因此：
- 从工作流首次成功运行起，逐日积累起报快照与观测；
- 运行报告从第一天起即有内容（数据窗口随时间变长）；
- **第一份覆盖完整自然月的月度报告，需积累约 1 个月**数据后才具备完整时效样本（早期月份的长时效/按天偏移样本会偏少，报告中以"样本不足"标注）。

## 扩展

- **新增站点**：在 `config/stations.yaml` 的 `stations` 下加一项（`id`/`name`/`lat`/`lon`/`obs_url`，`obs_url` 为 eia-data 对应"气象站基本信息"页的 URL 编码）。
- **新增模型**：Open-Meteo 模型直接在 `models` 下加入其模型名；中科天机模型在 `forecast/tianji.py` 的 `MODEL_SPECS` 登记映射（mode/production/factorCode）后再加入 `models`。
- **新增预报源**：实现 `weather_eval/forecast/base.py` 的 `ForecastProvider` 接口（返回统一结构的起报快照），在 CLI 中切换即可，评估/报告逻辑无需改动。已内置示例 `forecast/caiyun.py`（`CaiyunProvider`）：`fetch-forecast --source caiyun` 抓取，Token 取自 `CAIYUN_TOKEN` 环境变量；其返回的 `caiyun_v2_6` 模型已加入 `models`，故 `report` 自动纳入对比。和风天气同例（`forecast/qweather.py`，`QWeatherProvider`，模型名 `qweather_v1`）。中科天机同例（`forecast/tianji.py`，`TianjiProvider`）：因各模式最新可用起报轮次可能不同步，其 `fetch_snapshot` 返回**按模型独立的快照列表**（各自 issue_iso 与时间轴），CLI 已兼容 dict（共享时间轴，逐模型拆分）与 list（独立快照）两种返回形态。

## 目录结构

```
config/stations.yaml      站点、模型、评估参数
src/weather_eval/
  timeutil.py             北京时工具
  config.py  storage.py   配置与 JSON 存档（原子写/去重/幂等）
  obs/                     观测源（eia-data 抓取解析）
  forecast/                Open-Meteo / 彩云天气 / 和风天气 / 中科天机 快照器（base.py 抽象接口）
  evaluate.py             配对 + cyeva 指标 + 报告数据组装
  report/                  Jinja2 + ECharts 报告渲染
  __main__.py              CLI
data/                     观测档案 / 预报快照（git 跟踪）
reports/                  index.html 主报告 / monthly 月度归档 / vendor ECharts 本地副本（git 跟踪，部署 Pages）
tests/                    pytest（含 cyeva 手算对拍）
```

## 已知约束

- Python 必须为 **3.12.6**（或经验证可导入 cyeva 的 3.12 补丁）；3.13 不被 cyeva 支持。
- `pint` 锁定 **0.24.4**；若升级到其它版本可能导致 cyeva 导入失败或定义解析错误。
- eia-data.com 页面结构若改版，抓取解析（`obs/eia_data.py`）需相应调整（已有 `wd` JSON 与 HTML 表格双路回退）。
- Open-Meteo 免费档有速率限制，批量回溯请控制频率。
- **彩云天气（v2.6）接入约束**：
  - 需 `CAIYUN_TOKEN`；鉴权失败（如 `token is invalid`）会抛出清晰错误而非静默产出空数据。
  - **逐小时时间戳锚定在请求时刻**（形如 `2026-08-27T15:10+08:00`，分钟随请求波动且带时区偏移），而观测落在整点。提供方已**下取整到整点**再与观测配对（逐小时量级下分钟偏差可忽略），否则会与整点观测全部错配、丢失样本。
  - **长时效返回与 User-Agent 强相关**：本测试 Token 下，使用默认 `python-requests` UA 仅返回约 48 个逐小时点，而本项目固定 UA `weather-api-eval/0.1 (+https://github.com/)` 可返回完整 384 点（约 16 天）。提供方在返回点数明显少于请求时记录 WARNING 以暴露此类静默降级；**请勿随意改动该 UA**。
  - 彩云不做格点吸附，`location` 即请求坐标；响应该端点无 `elevation` 字段，快照中 `elevation` 记 `None`。
  - `fetch-forecast --source caiyun` 独立于 Open-Meteo，需手动运行（CI 默认 `all` 不含彩云，避免 Token 缺失导致失败）。彩云预报为"未来"时刻，故抓取当次即与历史观测 0 配对属正常；数值评估将在后续观测积累后自动填充。
- **和风天气（weather/v1）接入约束**：
  - 需 `QWEATHER_API_KEY`；建议同时设 `QWEATHER_API_HOST` 为控制台分配的专属 API Host——旧公共域名（devapi/api.qweather.com）自 2026 年起逐步停止服务，未设置 Host 时会退回 `devapi.qweather.com` 并记录 WARNING。API Key 经 `X-QW-Api-Key` 请求头传递，不出现在 URL 中。
  - **坐标契约：小数不超过 2 位**。站点坐标（4 位小数）在请求前取整到 2 位；快照中 `grid_lat/grid_lon` 记录实际参与查询的取整坐标，`requested_lat/lon` 保留原值。
  - **时间换算**：新版接口的 `forecastTime` 为 UTC（`Z` 结尾），提供方统一转换为北京时 naive 墙钟并**下取整到整点**后再与观测配对（同彩云口径）；旧版 v7 的 `fxTime` 已带 `+08:00` 偏移，同样处理。
  - **时效与降级**：默认请求未来 240 小时逐小时。若凭据/主机尚未开通 weather/v1 路由，会**自动改用旧版 `/v7/weather/{24|72|168}h`** 并告警（免费开发版仅开放 24h 档）；返回点数少于请求时同样以 WARNING 明示。订阅档位决定该源的实际覆盖时效，报告中长时效"样本不足"属正常现象。
  - 限速遵循官方建议对网络错误/5xx/429 做**指数退避**重试；鉴权/权限/参数类 4xx 不做无意义重试直接上抛，鉴权失败（401）给出可操作的错误信息。
  - 同样独立于 Open-Meteo 抓取（CI 中以 `QWEATHER_API_KEY` Secret 存在与否决定是否执行），降水为当小时累计毫米、温度摄氏度，与其他源同口径。
- **中科天机（网页接口抓取）接入约束**：
  - 接口为 `www.tjweather.com/vis/` 可视化页面的单点查询（`/meteorological/spas/single-point/query`），**非官方开放 API**：游客态无需鉴权，但该契约可能随页面改版变化（参数/要素码见 `forecast/tianji.py` 顶部 docstring，均经 2026-08 线上实测）；CI 中以 `continue-on-error` 可选步骤运行，失败不阻断主流程。
  - **起报轮次与回退**：北京时每天 08/20 时两轮；最新轮次有发布延迟且各模式进度不同步（对未发布轮次查询返回 200 但数据为空）。提供方逐模式向过去回退探测（最多 4 轮）取首个非空轮次，并按模式缓存（跨站点复用）。
  - **时间语义**：请求参数 `baseTime` 与响应 `forecastTimeString` 均为**北京时** `YYYYMMDDHH`（服务端回显的 ISO 时间为 UTC，恒差 8h）；逐小时序列从起报后 1 小时开始，起报当刻不在序列中。
  - **快照粒度**：各模式最新可用起报可能不同步，为避免 lead 分组被跨模式错位污染，`TianjiProvider.fetch_snapshot` 返回**按模型独立的快照列表**（各自 issue_iso），CLI 逐份存档（`save_forecast_snapshot` 按 站×模型×起报 幂等）。
  - **降水口径**：`pratesfc` 为逐小时降水率（mm/h），作为"前 1 小时累计"的近似与观测配对（同 Open-Meteo `precipitation` 假设）；公里级融合产品温度/降水使用不同产品网格码（`c1km`/`c2_5km`）。
  - **模型名对应**：`tj_km_fusion`=公里级融合（nextgen）、`tj_t2_early`=天机2/DA（T2-Early）、`tj_t2`=天机2/ND（T2）、`tj_t1`=天机1/ND（T1，其本身即 AI 驱动，站点无非 AI t1 轮次）、`tj_t1h_ai`=T1H-AI（T1-AI 高分辨率版）。
- **伏羲中期（FuXi-C88，网页接口抓取）接入约束**：
  - 接口为 `fuxi-ai.cn/visual/weather` 可视化页面背后的自有网关（`/gw/weather/api/v1/weather/queryWeatherTile` + `queryWeatherInfo`），**非官方开放契约**：游客态无需鉴权，但参数/结构可能随页面改版变化（契约细节见 `forecast/fuxi.py` 顶部 docstring，均经 2026-08 线上实测 + 前端 JS 逆向）；CI 中 `continue-on-error` 可选步骤。
  - **起报锚点是隐性契约**：点位响应只有 step 1..360、无绝对时间，起报时刻取自 tile 接口的 `startTime`（`YYYYMMDDHH`，**UTC 语义**，北京时 = +8h；由前端 `moment.utc().local()` 解析方式与"step1 辐射≈0"的实测共同佐证）。时刻 = 北京时起报 + step 小时。tile 与点位是两个接口，理论存在读到不同轮次的极小竞态，无从校验。
  - **数值口径**：响应值为字符串（须 float 化）；`t2m` 已是 ℃；`tp` 页面图例为 mm/h（逐小时降水率），作为"前 1 小时累计"的近似与观测配对（同 pratesfc 口径）。**注意**数据服务 `/models` 元数据把 c88 的 tp 标为 "Total precipitation, mm"——两条产品线口径可能不同，若日后对比发现该源日降水系统性偏大 ~6 倍，应优先复核此处。
  - **发布滞后**：c88 可视化产品线发布明显滞后（实测 15 时最新锚点仍是前一日 12Z 轮），且接口无起报参数、无法回退探测——属产品形态，非故障。
- **伏羲确定性（FuXi-Det，数据服务 API）接入约束**：
  - 需 `FUXI_DATA_TOKEN`（登录 `fuxi-ai.cn/fuxi-data` 页面后由页面换取的查询 Token，经 `Authorization` 头传递、**无 Bearer 前缀**）；401 时给出可操作错误信息，不会静默产出空数据。
  - **只接 FuXi-Det**：伏羲中期（FuXi-C88）必须走可视化接口（`--source fuxi`），两条产品线的接口、坐标网格与单位口径都不同，不得混接。
  - **起报探测**：`initTime/isAvail` 游客可用，`initTime` 只传 UTC 日期（`YYYY-MM-DD`），返回该日可用 UTC 小时列表；从 UTC 今天向过去回退最多 4 天，取首个非空日的最大小时。`queryWeatherInfo` 的 `initTime` 为 **UTC** `YYYY-MM-DD HH:00:00`（页面小时按钮带 "z" 后缀）。
  - **单位自适应**：响应 `units` 数组按 `var_names` 定位 t2m——声明 K 则减 273.15 转 ℃；未声明单位时告警并按 ℃ 处理，且数值普遍呈开尔文量级（>150）时额外触发口径漂移预警。降水 `tp` 单位 mm；**累计窗口官方未说明**，当前按"逐时刻值 ≈ 前 1 小时累计"配对，并内置单调性哨兵（若序列近乎单调不减——自起报累计的典型形态——会 WARNING 提示口径存疑）。
- **风乌（FengWu-GHR-9km，网页接口抓取）接入约束**：
  - 游客态可用但**服务端截断至起报后 166h、3 小时步长**（56 点）；填 `FENGWU_API_KEY`（`Authorization: Bearer` 头）解锁逐小时 360h。Key 无效（401）立即报错而非回退游客时效。
  - **时间分辨率处理**：温度对 3h 采样**线性插值**到逐小时（有 Key 时退化为恒等）；不外推首末采样之外。
  - **6 小时降水处理**：原生 `tp6h` 为截至该时刻的 6h 累计（窗口 (t-6h, t]，由 ssrd1h/ssr6h 并存与 ERA5 惯例推证）。因 3h 采样使相邻窗口重叠 3h，直接逐窗均摊会重复计总量——采用**相位平铺子集**（以首个采样为相位、每 6h 取一个窗口端点，窗口两两无缝拼接），每窗累计均摊 /6 到其 6 个小时：总量严格守恒（日降水 BIAS 不失真）、口径与 pratesfc 一致；代价是中间采样的信息被弃用、且 6h 均摊对 0.1mm 晴雨阈值偏保守（短时强降水被摊薄），属已知局限。展开口径在快照 `expansion` 字段留档。
  - 起报以 `availability.api_end_time`（最新可查起报）为首选，查询 400 时向过去逐轮（-6h）回退最多 4 轮；响应坐标为 9km 网格吸附值（作为 grid_lat/lon 留档）。
- **中科星图（GeoVis，官方 API）接入约束**：
  - 需 `GEVIS_TOKEN`（注册 + 开发者认证）；档位按 专业(120h)→进阶(48h)→基础(24h) 自动回退，实际档位记入快照 `tier` 字段（账号级缓存，跨站点复用）。
  - **时间语义**：`fc_time`/`start` 为 `yyyyMMddHH` 当地时间（Asia/Shanghai），直接按北京时处理；数据从查询时刻起报（`start` 即起报时刻），每天更新 7 次——同一抓取时刻重复查询会得到相同 start，按 issue 幂等去重。
  - **缺测语义**：官方异常值 999999（含 9999/99999 变体）→ None；`tem`=℃、`pre`=该小时降水量 mm（与观测 rain@t 直接同口径，无需近似）。
  - token 走 URL query 参数（官方契约），网络异常入日志前已做掩码，不泄漏凭据。

## 评估方法说明（关于样本与时效）

系统每次运行都**永久封存当时发布的起报快照**。评估时，对每一个（起报, 有效时刻）组合都是一条**独立样本**，并按其**实际时效 lead = 有效时刻 − 起报时刻**归入对应的「天桶 / 逐小时曲线 / 日偏移」。这是连续数值预报检验（continuous NWP verification）的标准做法：

- 同一个有效时刻会被多个历史起报共同覆盖，因此**短时效（1–24h）样本更多、长时效（如 15–16 天）样本相对少**——这是预期且正确的，时效曲线正是要反映"越临近准确率越高"的规律；
- 每个样本只计入一次（由起报快照唯一确定），不存在重复计数；
- 因此 `n` 较大是常态，`min_sample=5` 的"样本不足"只对长时效/长日偏移的早期月份生效。

> 观测来源（eia-data.com）返回的时间被当作**北京时（UTC+8，无夏令时）**处理，与 Open-Meteo 的 `timezone=Asia/Shanghai` 时间轴一致；配对基于绝对时间戳串，故不存在时区错位。

> **关于每天 3 次运行与起报快照**：Open-Meteo 的时间轴从"当地当日 00:00"起返回未来 16 天逐小时序列。因此按 `issue_iso`（时间轴首点）去重后，**每个（站点 × 模型 × 当日）只会归档一份快照**（当日首次成功抓取的那份；若该次失败，后续 13/20 点运行会自动补上）。三次日运行的更多价值在于：刷新观测、更早发现抓取异常、以及更频繁地出具报告。

## 报告系统重设计（2026-08）：对抗式审查发现并修复的问题

本次把"每次运行一份报告"重构为"单一累积主报告 + 月度归档"后，经通用子智能体对抗式审查，发现以下问题并已修复：

- **（P0）热力图颜色编码错位**：热力图 data 由三元组改为四元组（追加样本数 n）后未指定 `visualMap.dimension`，ECharts 默认取最后一个维度着色——颜色实际编码的是**样本数**而非准确率，翻车日会因样本多而显示绿色。已显式指定 `dimension:2`。
- **（P0）对比图空态卡死**："预报 vs 实况"切到无观测站点时用 `textContent` 清空容器，切回时 `echarts.init` 幂等复用已脱离 DOM 的旧实例，图表永久停留在"暂无数据"提示上。已改为 dispose + 恢复容器 + 重建实例，并用含"无观测站点"的合成数据在浏览器实测空态 ⇄ 有数据往复。
- **（P1）归档页时间语义误导**：月度冻结页的"数据更新至"显示的是生成时刻（如 8 月归档显示 8-27），已改为显示数据窗口末尾（`meta.end`），页脚保留墙钟生成时间。
- **（P1）归档页对比图结构性缺失预报线**：原实现取"全局最新快照"画预报线，月度归档时最新快照在归档月之后、与观测窗口零交集。已改为"对每个时刻取当时已发布的最新一版预报"（按起报升序覆盖 + 只用窗口结束前发布的快照），主报告与归档页的预报线都能全程覆盖 72h 窗口。
- **（P1）归档列表延迟与测试缺口**：`monthly` 命令现在归档后立即重建主报告，新月报当次部署即出现在首页页脚；补充了 ranking 无条件计算与 `write_monthly_report` 的回归测试，清理恒真断言。
- **（P2）注入面统一与原子写**：所有内联 `<script>` 的 JSON（含站点中文名等新增 blob）统一走 `</`→`<\/` 转义；index/归档落盘改为临时文件 + 原子 rename（防半文件被 CI 提交部署）。另修复 ECharts 加载失败提示横幅在 `<head>` 中执行时 `document.body` 为 null 导致横幅自身抛错的陈年问题（移入 DOMContentLoaded）。
- **（P2）文案与可访问性**：综合得分公式在排行榜注明口径（含温度误差换算分 `100−RMSE×5`）；冠军横幅措辞改为"提前 1 天以内（1~24 小时）"；样本不足的排名卡不再发奖牌；归档页皇冠显示"当月最准"；下拉框补 label 关联；明细表注明 n 的取值口径；工作流步骤名与新行为对齐。

## 对抗式审查已修复的问题

初版代码经三轮对抗式审查（正确性/统计、抓取鲁棒性、运维/渲染），已修复：

- **起报时刻口径**：`issue_iso` 改为取 Open-Meteo 响应时间轴首点（原取本地请求时刻，跨整点会偏移最多 1 小时，污染所有时效判定）。
- **观测合并**：`save_obs` 改为逐字段合并，缺测（None）不再覆盖已有有效值，避免有效观测被临时缺测覆盖。
- **抓取健康度**：页面 200 但解析到 0 条时视为失败并上抛，CLI 汇总失败数且以**非零退出码**结束（GitHub Action 因此标红）；表格回退只选"观测表"（按表头含气温+降水量签名），避免误抓预报表；编码改用 requests 探测。
- **存储健壮性**：预报快照改为**按月合并单文件 + fcntl 文件锁**（解决数千小文件的性能与并发丢失更新）；JSON 读取容错（损坏文件跳过不拖垮整体）；原子写权限修正为 0644。
- **报告渲染**：ECharts 改为**仓库内本地副本**（`reports/vendor/echarts.min.js`），解决中国大陆 CDN 不可达与单点故障；内联 JSON 做 `</` 转义防脚本注入；开启 Jinja autoescape（仅 JSON 块 `|safe`）；热力图缺样本不再显示为 0%；覆盖率 None 友好显示；各图表块加 try/catch。
- **依赖与 CI**：`cyeva==0.2.3` 用 `pip install --no-deps` 单独安装（绕过其误钉的 `pint==0.24.3`，否则与 `pint==0.24.4` 冲突且 0.24.3 在 3.12+ 无法导入）；`requirements.txt` 锁定 `numpy==2.1.2` 与兼容的 `pint==0.24.4` 并提供 cyeva 的真实运行依赖（pandas/scipy）；`pip install` 同时装 `requirements-dev.txt`；月度汇总改为**仅每月 1 号北京时 06:00 那次运行**触发（消除冗余重算）；提交前 `git pull --rebase` 防止 push 被拒；Python 锁定 3.12.6。

> 审查中一条"多快照重复计数"的判断经核实为**误报**：连续数值预报检验本就按（起报, 有效时刻）独立样本、按真实时效分组，并非重复计数（见上节）。

## 彩云天气接入：对抗式审查发现并修复的问题

本次新增 `forecast/caiyun.py` 与 CLI `--source caiyun` 后，经对抗式审查发现以下问题并已修复：

- **评估引擎空数组崩溃（系统级）**：`cyeva` 在样本数为 0 时会抛 `ArrayLengthNotEqualError`。原 `temp_metrics`/`precip_binary_metrics`/`precip_graded_ts` 仅在 `n < min_sample` 时早退，而 `min_sample=0` 的合法配置下空桶会直达 cyeva 崩溃。已在三处指标函数增加 `n == 0` 早退保护（`evaluate.py`），使"样本不足"在任何配置下都安全返回 `None`，不污染评估结果。
- **时间戳整点对齐（关键正确性）**：彩云逐小时时间戳锚定在请求时刻（带 `+08:00` 偏移、分钟随请求波动），与整点观测无法精确配对。提供方将时间戳解析为北京时后**下取整到整点**再配对；若不下取整，整点观测会全部错配、导致彩云评估恒为"样本不足"。已用单测锁定下取整与去偏移行为。
- **长时效静默降级（关键健壮性）**：经验证，彩云对该 Token 的返回长度与 `User-Agent` 强相关——默认 `python-requests` UA 仅返回约 48 点，固定 UA 才返回完整 384 点。此降级**无任何报错**，会静默丢失约 14 天样本。已在提供方固定该 UA，并在返回点数明显少于请求时记录 WARNING 暴露降级；并以单测锁定"请求 384 步 + 固定 UA"的请求契约。
- **Open-Meteo 误告警清理**：`models` 列表已含 `caiyun_v2_6`，原 `fetch-forecast`（Open-Meteo）会把它当作"响应缺失模型"刷警告。已在 Open-Meteo 分支过滤掉非其所属模型，消除噪声。
- **鉴权与异常路径**：`status=failed`（如 `token is invalid`）与 HTTP 错误均被捕获并转为清晰错误；缺 `CAIYUN_TOKEN` 时构造即报错；缺失 `temperature`/`hourly` 等结构时早退报错而非产出空快照。
- **长时效 lead 口径小偏差（已知、可接受）**：因时间戳下取整，`issue_iso` 取整点到小时，个别临近时效边界的样本 lead 可能被高估至多 ~59 分钟；与 Open-Meteo 同样取整到小时，口径一致，对天桶/逐小时曲线影响可忽略。

## 和风天气接入：对抗式审查发现并修复的问题

本次新增 `forecast/qweather.py` 与 CLI `--source qweather` 后，经对抗式独立审查发现以下问题并已处置：

- **（P0）降水解析错误导致数据静默全丢**：新版接口的带单位量纲统一为 `{value, unit}` 嵌套对象——`precipitation.amount` **本身是对象**（`{"value":0.09,"unit":"mm"}`），初版实现把它当数字 `float()` 必然失败归 `None`。后果是该源逐小时降水全部为缺测、晴雨/TS 指标静默缺失且无任何报错。已改为统一的 `_metric_value` 解析（嵌套对象与标量双形态兼容），并把**单测 mock schema 改为官方真实结构**、新增标量兼容用例锁定双形态（此前 mock 用了简化结构致测试"假绿"放行该缺陷）。
- **（P2）可选源步骤可拖垮主流程**：CI 中彩云/和风抓取位于核心 `all`（观测+报告+提交）之前，若 Key 已配但 Host 错/网络故障会令整个 job 失败、阻断主交付物。已为两个可选步骤加 `continue-on-error: true`（失败仅标注该步）；CLI 对独立源初始化失败改为清晰日志 + 非零退出而非裸 traceback。
- **（P3）v7 档位选择文档口径**：请求时效 <24h 时旧版档位只能取最小档 24h（多余点由快照层截断），docstring 已修正为准确描述。
- **（部分驳回）"401/403 一律快速失败不做降级"的建议未全盘采纳**：401 保持快速失败（账号级鉴权问题在两代端点必然同结果）；但 403 未纳入快速失败——免费凭据可能只是**尚未开通 weather/v1 路由**而 v7 可用，若贸然终止会错杀可降级场景。折中：403 触发降级的同时在 WARNING 中列出官方错误码的三类常见原因（额度不足/API Host 不符/无权限），若 v7 也失败则联合异常同时携带两侧状态码，根因仍可一步定位。
- 其余核查项确认无误：UTC→北京时换算与下取整口径、坐标 ≤2 位小数契约（lat 在前）、`X-QW-Api-Key` 仅经请求头传递且日志/异常脱敏、hours 截断告警、去重排序、Open-Meteo 分支对新模型的排除链、快照幂等存储、既有彩云/Open-Meteo 测试零回归。

## 中科天机接入与 Open-Meteo 扩容：对抗式审查发现并修复的问题

本次新增 `forecast/tianji.py`（中科天机 5 模型，网页接口抓取）与 Open-Meteo 9 个新模型（合计 12 个 OM 模型，config 共 19 个模型）后，经两路独立对抗式审查发现以下问题并已修复：

- **（P0）单模型失败拖垮整站**：`fetch_snapshot` 模型循环中任一模型抛错（产品下线、探测穷尽、瞬时网络错误）都会冲出循环，同站其余模型已抓到的快照全部丢弃，且"每轮只存最新可用起报"意味着丢失的轮次永久缺档。已改为 per-model 容错：失败的模型告警后跳过、返回成功子集，仅全部模型失败才上抛；已知失败模型在同次运行内跳过，避免后续站点重复 4 轮探测（每站最多浪费 16 次请求）。
- **（P0）抓取失败退出码恒 0**：`main()` 分发表丢弃命令返回值，单独运行 `fetch-forecast` 即使 4 站全失败也以 0 退出——CI 的 `continue-on-error` 步骤（彩云/和风/中科天机）连"失败标注"都不会出现，失败完全静默。已让返回的失败数反映到退出码（`sys.exit(1)`），与 `all` 的既有行为对齐。
- **（P1）降水产品线静默缺失且幂等锁死**：公里级融合的温度（`c1km`）与降水（`c2_5km`）是两条独立产品线，发布可能不同步；探测只用温度要素，降水为空时会静默存档一份降水全 None 的快照且无任何日志。已增加 WARNING（温度照常入库、该轮降水计为缺测、评估显示"样本不足"），并顺带增加"温度序列全部缺测"的契约漂移告警。
- **（P2）HTTP 500 + 业务错误码被当瞬时故障重试**：该服务把参数/产品类确定性错误包在 HTTP 500 + `{"code":11001,...}` 里返回，原实现一律按 5xx 退避重试 4 次。已改为：能解析出业务错误码（或 4xx）视为确定性失败直接上抛，真 5xx/网络错误才重试；最终错误附响应体摘要便于定位契约漂移；timeout 改为 (connect, read) 元组。
- **（P2）其他加固**：起报探测命中序列直接复用（省 1 次请求/模型）；回读校验响应 `baseTimeString` 与请求一致（防服务端"静默就近替换"导致 lead 错位）；缓存轮次意外空时作废重探；Open-Meteo `_model_key` 的裸键回退限定单模型请求（防多模型响应混入裸键时多模型共享同一数组）；config 无任何 tj_* 模型时在构造期即报错；`storage` 读取补齐 JSON 损坏容错（与模块文档承诺一致，损坏文件告警跳过不拖垮报告）；`ForecastProvider` 契约文档更新（dict 与 list[dict] 两种返回形态）。
- **（P2）报告可读性**：模型扩到 19 个后，逐日降水 TS 的 16 桶并排柱每柱仅数像素，改折线；热力图行标签限宽截断；页脚来源文案与实际模型清单对齐。
- **测试防"假绿"加固**：请求契约用例以硬编码的线上实测参数表（mode×production×factorCode 共 10 组）校验每次请求，`MODEL_SPECS` 映射漂移即红；另补探测穷尽、部分模型失败、降水空序列、月末/年末轮次边界、500 分类重试、CLI list 分支存档、Open-Meteo 裸键限定、storage 损坏容错等回归用例。
- **（已核查无误）**：TJ 独立快照与评估引擎的按目录读取/下标配对交互（跨模型起报不同步不污染 lead 分组）、`candidate_base_times` 的 08/20 时与跨天边界、同日两轮快照文件名无碰撞、cyeva 对含 NaN 对样本的剔除正确性（与手工掩膜对拍一致）、19 模型的标签/配色全覆盖且 JS 有兜底色。
- **（记录不修）**：Open-Meteo 12 模型单请求在任一模型名将来失效时会级联失败（当前 12 个名字经线上实测全部有效，属前瞻性风险）；明细指标表 19 模型横向滚动偏长；`best_match` 当前择优结果与 `ecmwf_ifs` 数值一致属数据源现象；快照文件线性增长的中期归档策略。

## 伏羲/风乌/中科星图接入：对抗式审查发现并修复的问题

本次新增 4 个预报源（`fuxi.py`/`fuxi_data.py`/`fengwu.py`/`geovis.py`，config 共 23 个模型）后，经独立对抗式审查（全部缺陷先实证复现再修复）发现并修复：

- **（P0）fuxi_data 毒化时间串可拖垮全链路报告**：`_parse_utc_ts` 无法解析的时刻原本会产出**空串时间键**入库（`['']` 是 truthy，防御拦不住），评估引擎 `parse_iso("")` 抛错且无容错——一条毒化快照即可让 `report`/`monthly`/`all` 对**所有站点所有模型**的报告生成永久崩溃（空快照还被存档幂等锁死）。已改为：时间解析升级为容忍毫秒/小写 z/零偏移（`fromisoformat` 统一路径），仍无法解析的**整列剔除**（时间轴与数值逐列对齐），并以单测锁定。
- **（P1）geovis 9999 缺测变体漏过**：`MISSING_TOL=99990` 拦不住国内气象常见的 9999 缺测码，9999℃ 会污染日最高温与 RMSE。已降阈值至 9999 并补用例。
- **（P1）geovis token 泄漏面**：token 走 URL query（官方契约），requests 网络异常消息携带完整 URL（含 token 明文）入日志（GitHub Actions 日志公开可见）。已对 `last_err`/body 摘要做统一掩码，单测断言日志无 token。
- **（P1）fengwu 401 被当"起报不可查"逐轮回退**：Key 无效是账号级错误，回退 4 轮毫无意义且最终错误消息误导（"最近 4 个起报轮次均不可查"）。已改为 401/其他非 400 的 4xx 立即上抛，仅 400 参与起报回退；单测断言查询次数为 1。
- **（P1）fuxi_data/geovis 4xx 被吞后无意义重试**：`_Rejected` 缺少专门分支、落入 `except Exception` 退避重试（4 站最多浪费 84s + 配额）。已补 `except _Rejected: raise`（与 fuxi.py 对齐），补 400-无重试用例。
- **（P1）fuxi_data 单位缺失静默按 ℃ 处理**：`units` 缺失时 K 值（≈300）不换算直接入库。已加"未提供单位"WARNING + 开尔文量级哨兵（未声明 K 但数值普遍 >150 → 口径漂移预警）。
- **（P1）fuxi/geovis 空时间轴快照静默入库并被幂等锁死**：响应非空但字段全非法时 `time=[]` 无异常通过，存档后同 issue 永久跳过、正常数据进不来。已在两处入库前校验 `time` 非空（fuxi_data/fengwu 原有防御），补回归用例。
- **（P2）加固**：fuxi_data 轮次取 `max(hours, key=int)`（不再依赖零填充字典序）；`isAvail` 无 msgCode 且无 data 的响应不再当"该日无数据"静默回退；fengwu 数值解析统一 NaN/inf/非法串 → None（与其余源同口径）、重复时刻去重统一保留首见、`parse_iso_z` 对非零时区偏移显式拒绝（防 +08:00 结尾被静默误读整体错 8h）。
- **（记录不修）**：fuxi tile 锚点缓存会把"tile↔点位轮次竞态"窗口从单站扩大到整批站点（进程生命周期短、概率极低，docstring 已留档）；各新源缓存无 tianji 式"作废重探"自愈（同上）；风乌游客态 3h 步长导致降水评估样本天然只有逐小时口径的 1/3 覆盖密度（产品形态）。
- **（已核查无误）**：fuxi UTC 锚点换算与跨月边界、fengwu 6h 降水平铺子集的总量守恒数学（逐点推演 + 守恒单测）、fuxi_data 起报探测跨日回退、geovis 档位回退与账号级缓存、CLI 分发/排除链/退出码、23 模型标签配色齐备、存档幂等无 issue 碰撞。
