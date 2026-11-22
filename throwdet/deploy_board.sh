#!/usr/bin/env bash
# Deploy a throw-motion detector to the Kria (bitstream + config + weights), with backup/restore.
#
#   throwdet/deploy_board.sh <model_dir> [firmware_dir]     # full deploy (first time / new firmware)
#   throwdet/deploy_board.sh <model_dir>                    # weights.txt + config.yaml only
#                                                           # (switch between models A-E2, same firmware)
#   throwdet/deploy_board.sh --restore <backup_name>        # put a backup back (e.g. the 11-class demo)
#   throwdet/deploy_board.sh --list                         # list the backups on the board
#
# firmware_dir holds spleat.bit.bin and spleat.dtbo (riversdk.dev firmware job, firmware-kv260.tar.gz
# unpacked). Two firmwares fit our models (same v1 architecture, potentials Q4.12):
#   2 outputs, weights Q3.5: riversdk job #43, models in throwdet/models/*
#   11 outputs, weights Q2.6: the gesture firmware (job #25 layout) shared with the rest of the team,
#     models exported with export.py --pad-outputs 11 --weight-frac 6 (e.g. D_person01_neg_11out)
# Without firmware_dir the script checks that the board's output count matches the model.
#
# Every deploy first copies the files it replaces to /root/fw_backup/<timestamp>/ on the board.
# Host: KRIA=root@192.168.77.1 (default) or KRIA=root@durance-kria.local.
set -euo pipefail
KRIA=${KRIA:-root@192.168.77.1}
FW=/usr/lib/firmware/xilinx/spleat
CFG=/root/.config/spleat-server/config.yaml

reload() {
  # Reload the FPGA app (only needed for a new bitstream) and restart the SPLEAT server.
  ssh "$KRIA" "set -e
    systemctl stop spleat-server.service
    if [ \"$1\" = fpga ]; then xmutil unloadapp >/dev/null 2>&1 || true; xmutil loadapp spleat; fi
    systemctl start spleat-server.service
    sleep 2
    systemctl is-active spleat-server.service
    grep -A3 '^outputs' $CFG | tr '\n' ' '; echo"
}

backup() {
  local name
  name=$(date +%Y%m%d_%H%M%S)
  ssh "$KRIA" "set -e; d=/root/fw_backup/$name; mkdir -p \$d
    cp $FW/spleat.bit.bin $FW/spleat.dtbo $CFG /root/weights.txt \$d/
    echo \"$2\" > \$d/NOTE
    echo backup: \$d" >&2
  echo "$name"
}

if [ "${1:-}" = --list ]; then
  ssh "$KRIA" 'for d in /root/fw_backup/*/; do echo "$(basename $d): $(cat $d/NOTE 2>/dev/null)"; done'
  exit 0
fi

if [ "${1:-}" = --restore ]; then
  b=/root/fw_backup/${2:?backup name, see --list}
  ssh "$KRIA" "set -e; test -f $b/spleat.bit.bin
    cp $b/spleat.bit.bin $b/spleat.dtbo $FW/
    cp $b/config.yaml $CFG
    cp $b/weights.txt /root/weights.txt"
  reload fpga
  echo "restored $b"
  exit 0
fi

MODEL=${1:?model dir, e.g. throwdet/models/D_e10_fold3}
FWDIR=${2:-}
for f in weights.txt config.yaml readout.json; do
  test -f "$MODEL/$f" || { echo "missing $MODEL/$f" >&2; exit 1; }
done
outs=$(sed -n '/^outputs/,$p' "$MODEL/config.yaml" | awk '/channels/{print $2; exit}')
wf=$(python3 -c "import json;print(json.load(open('$MODEL/readout.json'))['ptq_check']['weight_format'])")
n=$(wc -w < "$MODEL/weights.txt" | tr -d ' ')
echo "$(basename "$MODEL"): $outs outputs, weights $wf, $n values"
case "$outs/$wf" in
  2/Q3.5) ;;   # 2-class firmware, riversdk job #43
  11/Q2.6) ;;  # 11-output gesture firmware (job #25 layout); export.py --pad-outputs 11 --weight-frac 6
  *) echo "unknown combination: $outs outputs with $wf weights" >&2; exit 1 ;;
esac
if [ -z "$FWDIR" ]; then
  board_outs=$(ssh "$KRIA" "sed -n '/^outputs/,\$p' $CFG | awk '/channels/{print \$2; exit}'")
  if [ "$board_outs" != "$outs" ]; then
    echo "the board firmware has $board_outs outputs, $(basename "$MODEL") needs $outs: pass a firmware dir" \
         "(2 outputs: vendor/firmware/throwdet-2class-v1-q3.5) or use the ${board_outs}-output export" >&2
    exit 1
  fi
fi
if [ "$(tail -c1 "$MODEL/weights.txt" | xxd -p)" = 0a ]; then
  echo "$MODEL/weights.txt ends with a newline (the board rejects it)" >&2; exit 1
fi

if [ -n "$FWDIR" ]; then
  test -f "$FWDIR/spleat.bit.bin" -a -f "$FWDIR/spleat.dtbo" || { echo "need $FWDIR/spleat.bit.bin and spleat.dtbo" >&2; exit 1; }
  backup x "before deploy of $(basename "$MODEL") with firmware $(basename "$FWDIR")" >/dev/null
  scp -q "$FWDIR/spleat.bit.bin" "$FWDIR/spleat.dtbo" "$KRIA:$FW/"
  scp -q "$MODEL/config.yaml" "$KRIA:$CFG"
  scp -q "$MODEL/weights.txt" "$KRIA:/root/weights.txt"
  reload fpga
else
  backup x "before weights of $(basename "$MODEL")" >/dev/null
  scp -q "$MODEL/config.yaml" "$KRIA:$CFG"
  scp -q "$MODEL/weights.txt" "$KRIA:/root/weights.txt"
  reload server
fi
echo "deployed $(basename "$MODEL") to $KRIA"
echo "test: .venv/bin/python throwdet/live_view.py --model $MODEL --backend board"
