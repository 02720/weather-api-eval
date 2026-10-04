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
# 用法：
#   ./scripts/install.sh                # 装运行时 + 开发依赖
#   ./scripts/install.sh --runtime-only # CI 只跑采集/评估时用
#   ./scripts/install.sh --frozen       # 严格按锁文件（带哈希）装，用于可复现构建
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

RUNTIME_ONLY=0
FROZEN=0
for arg in "$@"; do
  case "$arg" in
    --runtime-only) RUNTIME_ONLY=1 ;;
    --frozen) FROZEN=1 ;;
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

if command -v uv >/dev/null 2>&1; then
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
  echo "→ 未找到 uv，改用 pip 两步安装"
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

echo "✓ 安装完成。验证： python3 -m pytest -m unit -q"
