# 天气预报 API 准确度评估系统

自动抓取多个来源的数值天气预报，与实景观测对比，用 [`cyeva`](https://github.com/caiyunapp/cyeva) 评估**温度**与**降水**两个指标的准确度，并生成可在线浏览的多模型对比报告。

> 设计原则：预报必须在实况出现之前**存档固定**，事后再与观测配对比对，才是有效的检验。系统每次定时运行都会抓取并永久封存当时发布的起报快照，后续实况到达后再做验证。

## 支持的来源

- **预报（Open-Meteo）**：`ecmwf_ifs`、`ncep_gfs_global`、`dwd_icon_global`（三个模型一次请求返回，后续可继续在 `config/stations.yaml` 增模型）。
- **观测（环境气象数据服务平台 eia-data.com）**：各气象站"气象站基本信息"页，服务端直出近 24 小时逐小时实况（气温、降水、气压、湿度、风）。
- **评估框架**：`cyeva 0.2.3`（温度 RMSE/MAE/MBE/±1°C·±2°C 准确率；降水 0.1mm 晴雨二分类准确率/POD/空报率 FAR/漏报率/TS/BIAS，及分级降水 TS）。

## 评估口径（与报告顶部说明一致）

- **逐小时**：按起报后时效分 1–16 个「天桶」；温度 RMSE/MAE/MBE/±1°C/±2°C 准确率；降水 0.1mm 晴雨二分类指标。另含 1–72h 逐小时 RMSE 曲线。
- **按天**：北京时自然日（00:00–24:00）聚合日最高/最低气温、日降水量；按"有效日 − 起报日"的日偏移 1–16 天分组，评估日最高/最低气温 ±2°C 准确率与日降水 TS。
- 所有指标附样本数 n；**n < 5 视为"样本不足"，不出结论**。
- 降水口径对齐假设：观测 `rain@t`（t−1h→t 累计）对应 Open-Meteo `precipitation@t`（前 1 小时累计），偏移量在配置中可改。

## 环境要求

- **Python 3.12.6**（重要：cyeva 0.2.3 + pint 0.24.4 在更高 3.12 补丁上有 dataclass 兼容问题，已验证 3.12.6 可用）。
- 网络访问：Open-Meteo（HTTPS）、eia-data.com（HTTP）。Open-Meteo 免费档约 1 万次/天，无需 key。

## 快速开始（本地）

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 抓取 4 站近 24h 观测
python -m weather_eval fetch-obs
# 抓取 3 模型起报快照（未来 16 天逐小时）
python -m weather_eval fetch-forecast
# 生成本月至今的运行报告
python -m weather_eval report
# 浏览：打开 reports/latest.html
```

常用命令：

| 命令 | 作用 |
|------|------|
| `fetch-obs` | 抓取观测并归档（按时间戳去重） |
| `fetch-forecast` | 抓取起报快照并归档（幂等） |
| `report` | 生成本月至今的运行报告 + 更新门户 `reports/index.html` |
| `monthly [--month YYYY-MM]` | 生成指定月份（默认上一自然月）的月度汇总报告 |
| `all` | `fetch-obs` + `fetch-forecast` + `report`（GitHub Action 调用） |

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
- **新增预报源**：实现 `weather_eval/forecast/base.py` 的 `ForecastProvider` 接口（返回统一结构的起报快照），在 CLI 中切换即可，评估/报告逻辑无需改动。

## 目录结构

```
config/stations.yaml      站点、模型、评估参数
src/weather_eval/
  timeutil.py             北京时工具
  config.py  storage.py   配置与 JSON 存档（原子写/去重/幂等）
  obs/                     观测源（eia-data 抓取解析）
  forecast/                Open-Meteo 快照器
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
- **依赖与 CI**：`requirements.txt` 锁定 `numpy==2.1.2`；`pip install` 同时装 `requirements-dev.txt`；月度汇总改为**仅每月 1 号北京时 06:00 那次运行**触发（消除冗余重算）；提交前 `git pull --rebase` 防止 push 被拒；Python 锁定 3.12.6（规避更高补丁上 cyeva+pint 的 dataclass 导入问题）。

> 审查中一条"多快照重复计数"的判断经核实为**误报**：连续数值预报检验本就按（起报, 有效时刻）独立样本、按真实时效分组，并非重复计数（见上节）。
