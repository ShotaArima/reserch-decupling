#!/usr/bin/env bash
set -u

# 失敗しても次へ進めたいなら set -e は付けない
# 逆に、1件でも失敗したら止めたいなら set -euo pipefail にする

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR" || exit 1

# 最新のリモート情報を取得
git fetch origin

# branch|label|command
EXPERIMENTS=(
  "codex/implement-experiment-plan-for-scenario-a|Experiment 11|uv run scenarios/scenario11_latent_probe/run.py"
  "codex/create-experiment-plan-for-scenario-12|Experiment 12|uv run scenarios/scenario12_input_assignment_role_swap_probe/run.py"
  "codex/generate-experiment-plan-for-scenario-13|Experiment 13|uv run scenarios/scenario13_same_input_vs_role_split_probe/run.py \ --window-size 14 \ --steps 120 \ --seed 42"
  "codex/generate-experiment-plan-for-scenario-14-z4b5tq|Experiment 14|uv run scenarios/scenario14_role_clarification/run.py"
  "codex/generate-experiment-plan-for-scenario-15|Experiment 15|uv run scenarios/scenario15_common_branch_strengthening/run.py"
  "codex/generate-experiment-plan-for-scenario-16|Experiment 16|uv run scenarios/scenario16_stock_common_vs_specific/run.py"
  "codex/create-experiment-plan-for-scenario-17|Experiment 17|uv run scenarios/scenario17_horizon_role_gap/run.py"
)

switch_branch() {
  local branch="$1"

  # すでにローカルブランチがあるならそれへ移動
  if git show-ref --verify --quiet "refs/heads/$branch"; then
    git switch "$branch"
  else
    # なければ origin から tracking branch を作る
    git switch -c "$branch" --track "origin/$branch"
  fi
}

for item in "${EXPERIMENTS[@]}"; do
  IFS="|" read -r branch label cmd <<< "$item"

  echo "=================================================="
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${label} を開始"
  echo "branch: ${branch}"
  echo "command: ${cmd}"
  echo "=================================================="

  if ! switch_branch "$branch"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${label} 失敗: ブランチ切替不可"
    continue
  fi

  if eval "$cmd"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${label} を終了"
  else
    status=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${label} 失敗: exit code=${status}"
  fi

  echo
done

echo "すべての実験ループが終了しました。"