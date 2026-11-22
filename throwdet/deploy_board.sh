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
# unpacked). One firmware serves all models in throwdet/models (same architecture, 2 outputs,
# weights Q3.5, potentials Q4.12); only weights.txt differs between them.
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
grep -q "channels: 2" <(sed -n '/^outputs/,$p' "$MODEL/config.yaml") || { echo "$MODEL/config.yaml: not a 2-output model" >&2; exit 1; }
wf=$(python3 -c "import json;print(json.load(open('$MODEL/readout.json'))['ptq_check']['weight_format'])")
if [ "$wf" != "Q3.5" ]; then
  echo "$MODEL has weights in $wf, the firmware expects Q3.5 (re-export with export.py --weight-frac 5)" >&2
  exit 1
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
