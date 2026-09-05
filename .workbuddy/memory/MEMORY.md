# 项目长期记忆 — 天气预报 API 准确度评估系统

## 工程约定（遵守）
- **src 布局未打包**：运行/测试前需 `PYTHONPATH=src`（pytest 已在 pyproject 配 pythonpath=["src"]）；
  解释器用项目内 `.venv/bin/python`（Python 3.12，cyeva 0.2.3 不支持 3.13）
- **存档先行（第一性原理）**：预报必须在实况出现前封存，永不修改；幂等键 = 站×模型×起报；
  严格模式不回填历史起报，样本随时间积累。新源首日入榜是 n=0"样本不足"属预期
- **绝不静默**：缺测归 None 绝不折算 0.0；契约漂移/降级/残缺一律 WARNING + 快照字段留档；
  确定性失败（4xx/结构漂移）不重试、立即熔断，可重试的（网络/5xx/429）才退避
- **可测性契约**：`__main__.py` 的独立源提供方必须是**零参 lambda 工厂**（晚绑定），
  写成类对象会破坏测试的 `monkeypatch.setattr(m, "XxxProvider", lambda: Fake())`
- **新增独立预报源**：只改 `__main__.py` 的 `SOURCE_SPECS`（模型集合+提供方一处登记）+
  config 模型名 + CI 步骤；provider docstring 必须像 tianji/accuweather/msn 那样
  留档接口契约（时间语义/字段口径/降级行为），评估与报告逻辑零改动
- **降水口径**：各源窗口方向假设都落在各自 provider 并留档（README 口径表是索引）；
  有累计字段时优先做守恒校验（如 MSN raAccu）
- **逐日预报补位**（2026-09 新增）：快照可选 `daily_time`/`daily` 块（契约在
  forecast/base.py）；按天轨道双轨——逐小时聚合优先，覆盖不足才用日产品，逐条记
  temp_src/rain_src；开关 `daily_source_fallback`。边界：绝不反推逐小时、观测门槛
  不放松、日界差 ~1h 只披露。接入参考 open_meteo._parse_daily；**接入前必须核实
  该源日产品的日界是否为北京时自然日**（中国天气网白天/夜间口径不可照抄，且其
  SSR day=N 分页只有被渲染那天的 daily 汇总可信）
- 单测假 session 用 `types.SimpleNamespace`（嵌套类闭包有坑），且配 `retries=0` 防退避拖慢

## 源接入状态（2026-09-05）
- 10 源 25 模型；MSN（msn.cn，底层中国天气网）为最新接入：
  SSR 内嵌 redux-data、day=1..10 逐日分片、`lastUpdated` 下取整做起报锚点、
  时效上限 9.3 天（天桶 1..10）、温度为整数℃（系统性 ±0.5°C 量化劣势，README 已披露）
- 观测源 eia-data 页面结构改版会影响解析（有双路回退）

## 环境
- 真实抓取验证要在**工作区内**建临时数据目录（WEATHER_EVAL_DATA_ROOT 指过去），
  /tmp 会被沙箱丢弃；探针脚本用完即删，勿留在仓库
