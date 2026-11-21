#!/usr/bin/env bash
# Run data generation / training on a remote GPU box over SSH.
#
#   tools/remote_gpu.sh setup user@host            # copy repo + dataset, install uv + deps
#   tools/remote_gpu.sh sync  user@host            # push code + dataset changes
#   tools/remote_gpu.sh run   user@host <command>  # run in the repo (detached tmux session)
#   tools/remote_gpu.sh fetch user@host            # pull runs/ (checkpoints, reports) back
#
# Optional: SSH_KEY=path/to/key  SSH_PORT=22
# The remote copy lives in ~/robotics_hackathon. vendor/downloads/riverops wheel is copied,
# the Durance dataset and recordings are not.
set -euo pipefail
cmd=${1:?command}; host=${2:?user@host}; shift 2
ssh_opts=(-o StrictHostKeyChecking=accept-new -p "${SSH_PORT:-22}"); [[ -n "${SSH_KEY:-}" ]] && ssh_opts+=(-i "$SSH_KEY" -o IdentitiesOnly=yes)
RSH="ssh ${ssh_opts[*]}"
DIR=robotics_hackathon

sync_up() {
  rsync -az -e "$RSH" \
    --exclude .venv --exclude runs --exclude 'vendor/*' --include 'vendor/downloads/riverops-*.whl' \
    --exclude "data/" --exclude ".git" --exclude "sim/" --exclude "dataset/build/synth_train" --exclude "videos/" --exclude "tools/models" \
    ./ "$host:$DIR/"
  $RSH "$host" "mkdir -p $DIR/vendor/downloads"
  rsync -az -e "$RSH" vendor/downloads/riverops-*.whl "$host:$DIR/vendor/downloads/"
}

case $cmd in
  setup)
    sync_up
    $RSH "$host" bash -s <<'EOF'
set -e
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
cd robotics_hackathon
command -v ffmpeg >/dev/null || (sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg tmux) || true
uv sync
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
nproc
EOF
    ;;
  sync) sync_up ;;
  run)
    $RSH "$host" "cd $DIR && mkdir -p runs && export PATH=\$HOME/.local/bin:\$PATH && tmux new -d -s job$$ '$* 2>&1 | tee runs/job$$.log' && echo started tmux session job$$, log runs/job$$.log" ;;
  fetch)
    mkdir -p runs/remote
    rsync -az -e "$RSH" "$host:$DIR/runs/" runs/remote/ --exclude cache ;;
  *) echo "unknown command $cmd"; exit 1 ;;
esac
