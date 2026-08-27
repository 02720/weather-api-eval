# 天气预报 API 准确度评估系统

自动抓取多个来源的数值天气预报，与实景观测对比，用 [`cyeva`](https://github.com/caiyunapp/cyeva) 评估**温度**与**降水**两个指标的准确度，并生成可在线浏览的多模型对比报告。

> 设计原则：预报必须在实况出现之前**存档固定**，事后再与观测配对比对，才是有效的检验。系统每次定时运行都会抓取并永久封存当时发布的起报快照，后续实况到达后再做验证。

## 支持的来源

- **预报（Open-Meteo）**：`ecmwf_ifs`、`ncep_gfs_global`、`dwd_icon_global`（三个模型一次请求返回，后续可继续在 `config/stations.yaml` 增模型）。
- **预报（彩云天气 Caiyun v2.6）**：模型名 `caiyun_v2_6`，Token 认证（环境变量 `CAIYUN_TOKEN`）。`fetch-forecast --source caiyun` 单独抓取，评估/报告逻辑与 Open-Meteo 完全一致。
- **预报（和风天气 QWeather weather/v1）**：模型名 `qweather_v1`，API Key 认证（环境变量 `QWEATHER_API_KEY`，专属 API Host 见 `QWEATHER_API_HOST`）。`fetch-forecast --source qweather` 单独抓取；请求未来最多 240 小时逐小时（免费档可能被限制为 24 小时，日志会明示），评估/报告逻辑与其他源完全一致。
- **观测（环境气象数据服务平台 eia-data.com）**：各气象站"气象站基本信息"页，服务端直出近 24 小时逐小时实况（气温、降水、气压、湿度、风）。
- **评估框架**：`cyeva 0.2.3`（温度 RMSE/MAE/MBE/±1°C·±2°C 准确率；降水 0.1mm 晴雨二分类准确率/POD/空报率 FAR/漏报率/TS/BIAS，及分级降水 TS）。

## 评估口径（与报告顶部说明一致）

- **逐小时**：按起报后时效分 1–16 个「天桶」；温度 RMSE/MAE/MBE/±1°C/±2°C 准确率；降水 0.1mm 晴雨二分类指标。另含 1–72h 逐小时 RMSE 曲线。
- **按天**：北京时自然日（00:00–24:00）聚合日最高/最低气温、日降水量；按"有效日 − 起报日"的日偏移 1–16 天分组，评估日最高/最低气温 ±2°C 准确率与日降水 TS。
- 所有指标附样本数 n；**n < 5 视为"样本不足"，不出结论**。
- 降水口径对齐假设：观测 `rain@t`（t−1h→t 累计）对应 Open-Meteo `precipitation@t`（前 1 小时累计），偏移量在配置中可改。

## 环境要求

- **Python 3.12.6**（cyeva 0.2.3 支持 3.10–3.12；`pint` 必须用 **0.24.4**，因为 cyeva 误钉的 0.24.3 在 3.12+ 上无法导入，见下方安装说明）。
- 网络访问：Open-Meteo（HTTPS）、eia-data.com（HTTP）、彩云天气 API `api.caiyunapp.com`（HTTPS）、和风天气 API `<你的专属 Host>.qweatherapi.com`（HTTPS）。Open-Meteo 免费档约 1 万次/天，无需 key；**彩云需 Token**，从环境变量 `CAIYUN_TOKEN` 读取；**和风需 API Key**，从环境变量 `QWEATHER_API_KEY` 读取。

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
# 抓取 3 模型起报快照（未来 16 天逐小时）
python -m weather_eval fetch-forecast
# 生成本月至今的运行报告
python -m weather_eval report
# 浏览：打开 reports/latest.html

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
```

常用命令：

| 命令 | 作用 |
|------|------|
| `fetch-obs` | 抓取观测并归档（按时间戳去重） |
| `fetch-forecast` | 抓取 Open-Meteo 起报快照并归档（幂等） |
| `fetch-forecast --source caiyun` | 抓取彩云天气 v2.6 起报快照（需 `CAIYUN_TOKEN`） |
| `fetch-forecast --source qweather` | 抓取和风天气起报快照（需 `QWEATHER_API_KEY`，建议同时设 `QWEATHER_API_HOST`） |
| `report` | 生成本月至今的运行报告 + 更新门户 `reports/index.html` |
| `monthly [--month YYYY-MM]` | 生成指定月份（默认上一自然月）的月度汇总报告 |
| `all` | `fetch-obs` + `fetch-forecast` + `report`（GitHub Action 调用，不含彩云/和风） |

## GitHub Actions 自动运行

1. 在 GitHub 新建仓库，将本项目推送（需包含 `data/` 与 `reports/`，已用 `.gitignore` 排除 `.venv` 等）。
2. 仓库 **Settings → Pages → Build and deployment → Source 选 "GitHub Actions"**。
3. 工作流 `.github/workflows/eval.yml` 已配置：
   - **定时**：北京时每天 **6:00 / 13:00 / 20:00**（cron `0 5,12,22 * * *` UTC）。
   - 每次运行抓取观测+预报、生成本次报告、提交数据/报告、部署 Pages。
   - **每月 1 号北京时**自动对该月做汇总，生成月度报告（也可手动 `workflow_dispatch` 指定 `month`）。
4. 报告在线地址：`https://<用户名>.github.io/<仓库名>/`（Pages 自动发布 `reports/` 目录）。

> 每次运行仅抓取近 24h 观测，因此单次失败不会丢数据（下次运行自动回补 24h 窗口）。

## 关于"首份完整月报"

系统采用**严格模式**（不在事前回填历史起报），因此：
- 从工作流首次成功运行起，逐日积累起报快照与观测；
- 运行报告从第一天起即有内容（数据窗口随时间变长）；
- **第一份覆盖完整自然月的月度报告，需积累约 1 个月**数据后才具备完整时效样本（早期月份的长时效/按天偏移样本会偏少，报告中以"样本不足"标注）。

## 扩展

- **新增站点**：在 `config/stations.yaml` 的 `stations` 下加一项（`id`/`name`/`lat`/`lon`/`obs_url`，`obs_url` 为 eia-data 对应"气象站基本信息"页的 URL 编码）。
- **新增模型**：在 `models` 下加入 Open-Meteo 支持的模型名（如 `cma_grapes_global`、`ukmo_global_deterministic_10km` 等）。
- **新增预报源**：实现 `weather_eval/forecast/base.py` 的 `ForecastProvider` 接口（返回统一结构的起报快照），在 CLI 中切换即可，评估/报告逻辑无需改动。已内置示例 `forecast/caiyun.py`（`CaiyunProvider`）：`fetch-forecast --source caiyun` 抓取，Token 取自 `CAIYUN_TOKEN` 环境变量；其返回的 `caiyun_v2_6` 模型已加入 `models`，故 `report` 自动纳入对比。和风天气同例（`forecast/qweather.py`，`QWeatherProvider`，模型名 `qweather_v1`）。

## 目录结构

```
config/stations.yaml      站点、模型、评估参数
src/weather_eval/
  timeutil.py             北京时工具
  config.py  storage.py   配置与 JSON 存档（原子写/去重/幂等）
  obs/                     观测源（eia-data 抓取解析）
  forecast/                Open-Meteo / 彩云天气 / 和风天气 快照器（base.py 抽象接口）
  evaluate.py             配对 + cyeva 指标 + 报告数据组装
  report/                  Jinja2 + ECharts 报告渲染
  __main__.py              CLI
data/                     观测档案 / 预报快照（git 跟踪）
reports/                  运行报告 / 月度报告 / 门户（git 跟踪，部署 Pages）
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

## 评估方法说明（关于样本与时效）

系统每次运行都**永久封存当时发布的起报快照**。评估时，对每一个（起报, 有效时刻）组合都是一条**独立样本**，并按其**实际时效 lead = 有效时刻 − 起报时刻**归入对应的「天桶 / 逐小时曲线 / 日偏移」。这是连续数值预报检验（continuous NWP verification）的标准做法：

- 同一个有效时刻会被多个历史起报共同覆盖，因此**短时效（1–24h）样本更多、长时效（如 15–16 天）样本相对少**——这是预期且正确的，时效曲线正是要反映"越临近准确率越高"的规律；
- 每个样本只计入一次（由起报快照唯一确定），不存在重复计数；
- 因此 `n` 较大是常态，`min_sample=5` 的"样本不足"只对长时效/长日偏移的早期月份生效。

> 观测来源（eia-data.com）返回的时间被当作**北京时（UTC+8，无夏令时）**处理，与 Open-Meteo 的 `timezone=Asia/Shanghai` 时间轴一致；配对基于绝对时间戳串，故不存在时区错位。

> **关于每天 3 次运行与起报快照**：Open-Meteo 的时间轴从"当地当日 00:00"起返回未来 16 天逐小时序列。因此按 `issue_iso`（时间轴首点）去重后，**每个（站点 × 模型 × 当日）只会归档一份快照**（当日首次成功抓取的那份；若该次失败，后续 13/20 点运行会自动补上）。三次日运行的更多价值在于：刷新观测、更早发现抓取异常、以及更频繁地出具报告。

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
