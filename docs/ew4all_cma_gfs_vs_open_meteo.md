# EW4ALL 的 CMA-GFS 与 Open-Meteo `cma_grapes_global` 一致性检验（留档）

**结论：不一致。** 两边是同一个模式系统（CMA-GFS/GRAPES-GFS 全球）、同一次起报、同一张 0.125°
格网、连**初始场都同到量化精度**，但 lead≥6h 之后的预报场并不是同一份数据——温度/湿度/降水的
数值差异达到本地网格尺度的 3–6 倍，重采样解释不掉。所以：可以认定"同一个模型"，
**不能**认定"同一条数据"（详见 §7）。

- 检验时间：2026-10-07 11:46 ~ 12:10 UTC
- 检验轮次：EW4ALL `dataTime=2026100700`（= 2026-10-07 00Z）↔ Open-Meteo 当时提供的那一轮
- 对齐口径：两边都用 **UTC**、`temporal_resolution=native`（3 小时步长）、lead 0 = 起报时刻
- EW4ALL 侧接口参数（前端 bundle 逆向确认）：`DataType` 枚举里 `CMA → "GRAPESGLOBAL"`，
  页面显示名 `model.CMA = "CMA-GFS"`，故 `data_type`/`mode` 取 **GRAPESGLOBAL**；
  可用轮次 00/12 UTC，`modelTimeList` 保留近 14 轮

---

## 1. 元信息对照

| 项 | EW4ALL CMA-GFS | Open-Meteo `cma_grapes_global`（native） |
|---|---|---|
| 标识 | `data_type`/`mode` = `GRAPESGLOBAL` | `models=cma_grapes_global&temporal_resolution=native` |
| 起报轮次 | 00 / 12 UTC | 00 / 06 / 12 / 18 UTC（API 只服务最近一轮；源码 `forecastHours`：06Z 只 120h） |
| 原生步长 | 3 h | 3 h（源码 `dtSeconds = 3*3600`）✅ 一致 |
| 时间戳语义 | `Datetime` = UTC，首点 = lead 0 | `timezone=UTC` 时同为 UTC，首点 = lead 0 ✅ 一致 |
| 时效覆盖 | 81 点 = 0–240 h（10 天，全程有值） | 本次仅 43 点 = 0–126 h；20 分钟后仍 43 点。本仓库归档的 7 份快照也只有 101–125 个非缺测小时 |
| 网格 | 0.125° 栅格；等值台阶**边界**恰落在 Open-Meteo 的格点上 → 其采样点相对 OM 整体偏移半格（0.0625°） | 原始模式网格 `RegularGrid(nx=2880, ny=1440, latMin=-89.9375, dx=dy=0.125)`，取最近格点（`interpolation=nearest_grid` 与默认结果相同） |
| 降水要素 | `TPE`＝**自起报累计**；逐窗口量另有 `HOURTPE`(3h)/`SIXTPE`(6h)/`TWELVETPE`(12h)/`DAYTPE`(24h) | 逐步（已 de-accumulate）的 3h 累计量 |
| 其他可用要素 | `TEM` `RHU` `GUST` `PTYPE` `GPH` `THICK` `PWAT`（`SLP`/`MSLP`/`U10`/`V10`/`C02` 对本模式为空） | 常规地面 + 部分气压层变量 |

## 2. 数值对照（同一轮次、同一 UTC 时刻、lead 0–126 h，温度单位 ℃）

| 点位（请求坐标） | OM 吸附格点 | mean\|Δ\| | max\|Δ\| | 完全相等 |
|---|---|---|---|---|
| wuzhou 23.4783,111.304 | 23.4375,111.25（103 m） | 0.281 | 0.90 | 5/43 |
| bobai 22.2981,109.99 | 22.3125,110.0（78 m） | 0.735 | 2.20 | 0/43 |
| pingnan 23.5431,110.419 | 23.5625,110.375（54 m） | 0.200 | 0.80 | 4/43 |
| wanning 18.8006,110.327 | 18.8125,110.375（39 m） | 0.344 | 1.60 | 3/43 |
| guangzhou 23.129,113.264 | 23.1875,113.25（21 m） | 0.302 | 1.80 | 6/43 |
| hanoi 21.03,105.85 | 21.0625,105.875（24 m） | 0.760 | 4.90 | 2/43 |
| manila 14.6,120.97 | 14.5625,121.0（10 m） | 1.588 | 5.30 | 2/43 |
| taiwan 23.5,120.5 | 23.5625,120.5（46 m） | 0.967 | 2.20 | 0/43 |
| bengal 22.0,90.0 | 22.0625,90.0（3 m） | 0.295 | 1.00 | 8/43 |
| **海面** 18.0625,112.0 | 同点（DEM=0） | 0.279 | 0.80 | 3/43 |
| **海面** 16.5625,118.5 | 同点（DEM=0） | 0.163 | 0.50 | 9/43 |
| **海面** 20.3125,116.125 | 同点（DEM=0） | 0.335 | 0.70 | 3/43 |

相对湿度同样不一致（EW4ALL `RHU` ↔ OM `relative_humidity_2m`，40 个重合时刻）：
海面 mean\|Δ\| = 1.03 pp（exact 2/40）、万宁 2.37 pp、孟加拉 2.05 pp。

## 3. 差异不能由"格点吸附"或"时钟"解释（机制判别）

1. **不是取错相邻格点。** 海面点 (18.0625,112.0) 上 DEM=0（Open-Meteo 不做高程订正），
   同坐标直接比：mean\|Δ\| = 0.273 ℃。而 OM 该格点与它 0.125° 相邻格点（N/S/E/W/NE/SW）
   的序列差只有 **0.058–0.117 ℃**。即 EW4ALL 的偏离量是本地网格尺度的 2–4 倍。
2. **也不是双线性重格。** 把 EW4ALL 值与 OM 四个角点值、以及"四角均值（双线性）"分别比：
   四角 0.287 / 0.295 / 0.312 / 0.320 ℃，四角均值 0.301 ℃（exact 1/40）——都比"OM 自身
   相邻点差异 0.06–0.12 ℃"大一个量级，两个假设都不成立。
3. **时钟是对齐的。** 把 EW4ALL 序列整体平移与 OM 比（wuzhou）：0 h → 0.295，
   ±3 h → 2.04/2.07，±6 h → 3.73/3.83，±24 h → 0.82/1.01。零平移最优，所以不存在
   EW4ALL 其他模式那种 UTC/北京时 8 小时坑（`Datetime` 与 `dataTime` 同为 UTC，与
   `ew4all.py` 既有契约一致）。
4. **Open-Meteo 侧还叠了一层高程订正。** 同一吸附格点 (23.4375,111.25)、不同请求点 DEM
   （59 / 28 / 5 m）返回的 lead-0 温度是 21.4 / 21.6 / 21.8 ℃（≈0.74 K/100m 直减率订正），
   序列并不相同；EW4ALL 没有这一环节。这是陆地上偏差普遍偏大（bobai 0.735、manila 1.588）
   的部分原因，但**不是**差异的全部来源——海面零高程处仍差 0.16–0.34 ℃。
5. **降水是两种量。** EW4ALL `TPE` 在 2 个轮次 × 4 个点位上都是**严格单调不减**、首步恒为
   0.0、末步 112.7 / 147.8 mm 等 → 它是"自起报累计"，不能像 `cma_ndfs` 的 `ONETPE` 那样
   透传入库。其逐 3 小时差分与 EW4ALL 自家的 `HOURTPE` 逐点吻合（≤0.1 mm 舍入）。
   用 `HOURTPE` 与 OM 的 native 降水比：孟加拉 mean\|Δ\| = 0.086 mm（38/42 相等，多为 0）、
   马尼拉 1.01 mm（总量 17.8 vs 27.6 mm）、万宁 2.32 mm（单步最大 13.8 mm；总量 102.9 vs 178.0 mm）。

## 4. 对照实验：这不是 CMA 专属问题

用同一套方法比 EW4ALL 的其他全球模式与 Open-Meteo 同名模型（海面点，run 2026100700，
native，UTC 对齐）：

| EW4ALL `mode` | Open-Meteo `models` | 重合点 | mean\|Δ\| | max | exact |
|---|---|---|---|---|---|
| `NCEP`（NCEP-GFS） | `ncep_gfs_global` | 120 | 0.162 / 0.239 ℃ | 0.8 / 1.7 | 28/120、18/120 |
| `D1D`（ECMWF-IFS） | `ecmwf_ifs` | 37 | 0.159 / 0.192 ℃ | 0.7 / 0.8 | 8/37、9/37 |
| `EDZWH`（ICON） | `dwd_icon_global` | 14 | 0.314 / 0.236 ℃ | 1.6 / 1.7 | 3/14、1/14 |

→ 两个平台对**所有**模式都各自做了重采样/后处理，数值不通用。EW4ALL 的时效反而普遍更长
（GRAPESGLOBAL 81 点 vs OM 43 点；NCEP 209 点），而 Open-Meteo 的原生步长更细（NCEP/ECMWF 逐 1 h）。

## 5. 处置建议

- **不接入为 `cma_grapes_global` 的替代/校验源**，也不做"两源取均值"——那会把两套不同的
  后处理当成同一模式的两次观测，破坏"同模式同网格代表点"的评估前提。
- 若要接入，登记独立模型名（如 `cma_gfs_ew4all`），并在 `ModelSpec` 里如实声明：
  `native_step_hours=3`、`expected_points=81`（0–240 h）、轮次 00/12 UTC；
  降水**必须**用 `HOURTPE`/3h（或 `TPE` 相邻差分），**不能**把 `TPE` 当逐时/逐步量透传
  （它是自起报累计，直接入库会把降水放大到 7000+ mm 量级）。
- 榜单解读时注意覆盖差：EW4ALL 到 240 h，Open-Meteo 本轮只到 ~126 h（且随灌入进度波动），
  分时效榜在 5 天以上的桶里两源样本构成不同。

## 6. 复现方式

检验脚本未入库（一次性探测），核心调用如下。

**接口坑（实测）**：`findByPoint` 的 `point` 传多个坐标时只返回**最后一个点**的序列
（响应里没有经纬度回显，无法区分取的是哪一点），逐点比较必须一站一发。

```bash
# EW4ALL CMA-GFS：3 小时步长、UTC、81 点（0–240h），TPE 为自起报累计
curl -s -X POST "http://ew4all.wmc-bj.net/EW4ALL/api/raster/findByPoint" \
  -H 'Content-Type: application/json' \
  -d '{"mode":"GRAPESGLOBAL","elements":"TEM","point":[[111.304,23.4783]],
       "projection":4326,"dataTime":"2026100700","level":0}'

# 模式代号来自前端枚举：assets/forecast-DtRFiizs.js 里 DataType {CMA_GDFS:"GDFS5KM", CMA:"GRAPESGLOBAL", EC:"D1D", ICON:"EDZWH", NCEP:"NCEP", JMA:"JAPGLB", FENGQING:"NMCFENGQING", ...}
curl -s "http://ew4all.wmc-bj.net/EW4ALL/api/modelTimeList?data_type=GRAPESGLOBAL&element=TEM"

# Open-Meteo：native = 3 小时步长，返回的 latitude/longitude 即吸附后的模式格点
curl -s "https://api.open-meteo.com/v1/forecast?latitude=23.4783&longitude=111.304
&hourly=temperature_2m,precipitation&models=cma_grapes_global
&temporal_resolution=native&timezone=UTC&forecast_days=11"
```

Open-Meteo 侧网格/步长/时效的事实取自其源码 `Sources/App/CMA/CmaDomain.swift`
（`RegularGrid(nx:2880, ny:1440, latMin:-89.9375, dx:0.125, dy:0.125)`、`dtSeconds=3*3600`、
`forecastHours = (run % 12 == 6) ? 120 : 240`）与 `CmaDownloader.swift`
（逐 3 小时 GRIB 文件 `NWPC-GRAPES-GFS-GLB-fNNN.grib2`，降水 `deaccumulateIfRequired`，
边下边 `finalise` 写出）。"边下边写"与本次看到的时效截断（43 点）在机制上相符，但属**推断**：
20 分钟后复查点数未变，未做跨时长的连续观测来坐实，也可能是上游文件本身只出到 f126。

---

## 7. 追问：能不能认为它们是"同一个模型"？

**同一个模式系统、同一轮次、同一初始场；但不是同一份预报数据。**

先补一个接口事实：`findByPoint` 的 `level` 不是层索引，而是**气压值**——
`GRAPESGLOBAL` 下 `level=0` 为地面场，`level=925/850/700/500/250/200` 可取等压面
（`300/150/100` 为空）。这让"用大尺度平滑量判别是否同一模式场"成为可能。

6 个海面点、run 2026100700、UTC、native 3h 步长：

| 量 | lead 0 原值（EW ↔ OM） | mean\|EW−OM\| 随时效（h：0/9/24/48/72/96） | OM 自身相邻 0.125° 格点差 |
|---|---|---|---|
| 500 hPa 位势高度 (gpm) | 5914.08837890625 ↔ 5914.0；5912.38818359375 ↔ 5912.0；5910.98828125 ↔ 5911.0 | 0.23 / 3.34 / 2.49 / 1.02 / 1.34 / 1.85 | 0.42–0.55 |
| 2 m 温度 (℃) | 28.4 ↔ 28.5；29.9 ↔ 29.9；29.6 ↔ 29.6 | 0.10 / 0.22 / 0.37 / 0.28 / 0.33 / 0.28 | — |

- **初值相同**：lead 0 的差（GPH 0.09–0.39 gpm、T2m −0.1~+0.5 ℃，多数为 0.0）落在 OM 的
  量化精度（GPH 1 gpm、温度 0.1 ℃）之内 → 同一模式系统的同一次起报；而且是在**同一坐标、
  同一 0.125° 格点**上取到同一数值，说明"§1 里的半格偏移"只是栅格原点写法差异，不是两套网格。
- **后续时效不同**：500 hPa 位势高度是极平滑的量，OM 相邻格点只差 0.4–0.6 gpm，而两源之差
  稳定在 1–3 gpm（单步最大 12.5 gpm）——是本地网格尺度的 3–6 倍，重采样/双线性/量化都造不出来。
  所以 lead≥6h 的场并非同一份预报输出（版本/后处理/数据 cut 不同，从外部无法再细分）。
- **但共享同一个信号**：逐点相关 r = 0.996（梧州）、0.971（万宁）、0.754（海面，因海面 T2m
  本身 σ 仅 0.4 ℃ 所以偏低）；两源之差相当于信号标准差的 9% / 20% / 68%。
- **放在本项目尺度上看**：2026-09 榜 `cma_grapes_global` 温度 MAE = 1.19 ℃、RMSE = 1.54 ℃，
  而两源之差 0.2–0.7 ℃ ≈ 模式自身误差的 1/5~1/2 → 互换不致翻转名次，但会引入系统性差异，
  且 5 天以上只有 EW4ALL 有数据（覆盖不对等）。

### 处置口径（一句话）

按"**同一个模式的两个产品实现**"处理：

1. 不能合并、不能互相替代、不能互为交叉校验真值（数值、降水口径、时效覆盖都不同）。
2. 不能当两个**独立**模式参与"跨模式择优 / 简单集成 / 模式多样性"类统计——同一物理内核 +
   同一初始场，样本强相关，并列等于把同一套物理计两次。
3. 若要接入：两个条目都保留，在 README 口径表里显式标注"同源模式、不同加工"
   （`cma_grapes_global` 与 `cma_gfs_ew4all` 归到同一"模式族"），凡涉及"多模式独立性"的
   指标按模式族去相关（同族合并计权或只取其一）。
