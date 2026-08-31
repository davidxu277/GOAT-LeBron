#!/usr/bin/env bash
# 本地跑一场 —— 从 .env 读密钥，先自检再开跑。
#
#   ./run_local.sh                       正式跑（Agent 自己写代码）
#   ./run_local.sh fm_baseline.yaml      官方 FM 基线复现
#   ./run_local.sh --check               只做自检，不开跑
set -euo pipefail
cd "$(dirname "$0")"

[ -f .env ] || { echo "❌ 没有 .env —— 先 cp .env.example .env 再填密钥"; exit 1; }
set -a; . ./.env; set +a

: "${DEEPSEEK_API_KEY:?❌ .env 里没有 DEEPSEEK_API_KEY}"
[ "${AGENT_PROVIDER:-}" = "deepseek" ] || { echo "❌ .env 里 AGENT_PROVIDER 必须是 deepseek"; exit 1; }

echo "提供商 : ${AGENT_PROVIDER}"
echo "型号   : ${AGENT_DEEPSEEK_MODEL:-deepseek-v4-flash（默认）}"
echo "密钥   : ${DEEPSEEK_API_KEY:0:6}…${DEEPSEEK_API_KEY: -4}（长度 ${#DEEPSEEK_API_KEY}）"
echo

echo "① 知识库自检（不花钱）"; python3 -m agent.cli check
echo
echo "② 配置能不能加载"
CONFIG="${1:-kuairand_task.yaml}"
[ "$CONFIG" = "--check" ] && CONFIG="kuairand_task.yaml"
python3 - "$CONFIG" <<'PY'
import sys, pathlib
sys.path.insert(0, "kuairand_goat_bridge/src")
from kuairand_bridge.goat_run import load_task
c = load_task(f"kuairand_goat_bridge/configs/{sys.argv[1]}")
print(f"   ✅ {sys.argv[1]} · trainer={pathlib.Path(c['trainer']).name} "
      f"· 数据存在={pathlib.Path(c['data_dir']).is_dir()} "
      f"· 输出={pathlib.Path(c['output_dir']).name}")
PY

echo
echo "③ 用假成绩单点一次医生 —— 这一步会真调 DeepSeek，验证密钥和型号"
python3 -m agent.cli doctor

if [ "${1:-}" = "--check" ]; then echo; echo "自检完毕，未开跑。"; exit 0; fi

echo
echo "④ 正式开跑（上限 50 轮 / 6 小时，中途可 Ctrl+C）"
cd kuairand_goat_bridge
python3 -m kuairand_bridge.cli run "configs/$CONFIG"
