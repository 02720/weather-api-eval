#!/usr/bin/env bash
# 一键安装（TASK-05）：把"两步 + 22 行注释说明"收敛成一条命令。
#
# 为什么需要这个脚本（而不是把注释留在 requirements.txt 顶部）：
# cyeva 0.2.3 的元数据把 pint 钉死为 ==0.24.3，而 pint 0.24.3 在 Python 3.12+
# 上会因 @dataclass(frozen=True) 的继承问题**导入失败**；实际可用的是 0.24.4。
# pip 的解析器不允许「cyeva==0.2.3 + pint==0.24.4」同时出现，所以既不能把 cyeva
# 直接写进依赖里一次性解析，也不能让每个新来的人自己读注释再手敲两步。
#
# 两条路，本脚本都支持：
#   * uv（推荐）：pyproject 里的 `override-dependencies` 让解析器一次通过；
#   * pip：先 --no-deps 装 cyeva，再装其余依赖。
#
# ⚠️ 两条路装进的**不是同一个解释器**，这是本脚本最容易骗人的地方：
#   * `--pip` 装进**当前解释器**（`python3`）——CI 用这条；
#   * uv 路径的 `uv sync` 装进**项目虚拟环境** `./.venv`，当前 `python3` 看不见它。
# 于是"装完了"与"python -m pytest/ruff 能跑"是两件事。2026-10-04 的 CI 红灯正是
# 栽在这里：CI 跑的是 `uv sync`（若镜像带 uv）或 pip 缺 ruff 的清单，随后
# `python -m ruff` 却问 setup-python 的解释器要模块。
# 两道防线：① CI 用 `--pip` 显式选路，不依赖镜像里"恰好有没有 uv"；
#          ② 脚本末尾做**真实自检**（不是 echo 一句建议），装没装上当场硬失败。
#
# 用法：
#   ./scripts/install.sh                # 装运行时 + 开发依赖（有 uv 就用 uv）
#   ./scripts/install.sh --pip          # 强制 pip 装进当前解释器（CI 用这条）
#   ./scripts/install.sh --runtime-only # CI 只跑采集/评估时用
#   ./scripts/install.sh --frozen       # 严格按锁文件（带哈希）装，用于可复现构建
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

RUNTIME_ONLY=0
FROZEN=0
FORCE_PIP=0
for arg in "$@"; do
  case "$arg" in
    --runtime-only) RUNTIME_ONLY=1 ;;
    --frozen) FROZEN=1 ;;
    --pip) FORCE_PIP=1 ;;
    *) echo "未知参数: $arg" >&2; exit 2 ;;
  esac
done

# Python 版本门禁：cyeva 0.2.3 在 3.13 上会 KeyError（PEP 667 改变了 exec/locals）
PY=$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo "unknown")
case "$PY" in
  3.12) ;;
  *) echo "⚠️  检测到 Python $PY：本项目需要 3.12（cyeva 0.2.3 在 3.13+ 上无法运行）。" >&2
     echo "    若你确知自己在做什么，可自行改用 3.12 后重跑本脚本。" >&2 ;;
esac

USE_UV=0
if [ "$FORCE_PIP" = "1" ]; then
  echo "→ --pip：强制 pip 装进当前解释器（不依赖镜像里有没有 uv）"
elif command -v uv >/dev/null 2>&1; then
  USE_UV=1
fi

if [ "$USE_UV" = "1" ]; then
  echo "→ 使用 uv（依赖冲突由 pyproject 的 override-dependencies 解决）"
  if [ "$FROZEN" = "1" ]; then
    uv sync --frozen --all-extras
  else
    if [ "$RUNTIME_ONLY" = "1" ]; then
      uv sync
    else
      uv sync --extra dev
    fi
  fi
else
  echo "→ 使用 pip 两步安装（装进当前解释器 $(command -v python3)）"
  if [ "$FROZEN" = "1" ] && [ -f requirements.lock.txt ]; then
    echo "   cyeva 无法参与 --require-hashes（它必须 --no-deps），先单独装："
    python3 -m pip install --no-deps cyeva==0.2.3
    python3 -m pip install --require-hashes --no-deps -r requirements.lock.txt
  else
    echo "   第 1 步：cyeva（--no-deps，跳过其错误的 pint 约束）"
    python3 -m pip install --no-deps cyeva==0.2.3
    echo "   第 2 步：其余运行时依赖（显式提供兼容的 pint==0.24.4）"
    python3 -m pip install -r requirements.txt
    if [ "$RUNTIME_ONLY" != "1" ]; then
      python3 -m pip install -r requirements-dev.txt
    fi
  fi
  echo "   第 3 步：本项目本体（可编辑安装，提供 weather-eval 命令）"
  python3 -m pip install -e . --no-deps || \
    echo "   （可编辑安装失败，不影响 PYTHONPATH=src 的用法）"
fi

# ------------------------------------------------------------------ 安装自检
# 为什么必须是真检查而不是一句 echo：安装脚本最恶劣的失败模式是"报告成功、实际
# 什么都没装进你要用的解释器"——本项目已经为此付过一次 CI 红灯（No module named
# ruff）。自检用**与后面步骤相同的解释器**去问工具要版本，问不到就当场非零退出，
# 把故障钉在安装这一步，而不是让它漂到三条命令之后以一句莫名其妙的报错出现。
if [ "$USE_UV" = "1" ]; then
  # uv sync 的目标是项目 .venv，当前 python3 看不见它——问错解释器等于没问
  CHECK_CMD=(uv run python)
  VERIFY_HINT="uv run pytest -m unit -q   # 或先 source .venv/bin/activate"
else
  CHECK_CMD=(python3)
  VERIFY_HINT="python3 -m pytest -m unit -q"
fi

selfcheck_fail() {
  echo "✗ 安装自检失败：$1" >&2
  echo "  这不是代码的问题，是依赖没装到位——**不要**继续往下跑，"
  echo "  那样只会在更远的地方看到更难懂的报错。" >&2
  exit 1
}

"${CHECK_CMD[@]}" -m pytest --version >/dev/null 2>&1 || \
  selfcheck_fail "pytest 不可用（${CHECK_CMD[*]} -m pytest）。"
if [ "$RUNTIME_ONLY" != "1" ]; then
  # ruff 是 CI 的硬门禁；缺它会让 ci.yml 的 lint 步骤以 'No module named ruff' 变红
  "${CHECK_CMD[@]}" -m ruff --version >/dev/null 2>&1 || \
    selfcheck_fail "ruff 不可用（${CHECK_CMD[*]} -m ruff）。它在 requirements-dev.txt 里，检查该文件是否被安装。"
fi

echo "✓ 安装完成并通过自检（$("${CHECK_CMD[@]}" -m pytest --version 2>&1 | head -1)）。"
echo "  验证： $VERIFY_HINT"
