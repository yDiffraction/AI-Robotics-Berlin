#!/bin/bash
# Start go2_dodge.py on the Kria in the background (log: /root/dodge/dodge.log). Extra args go to --ros-args,
# e.g. ./start.sh -p dry_run:=true
cd /root/dodge
/root/dodge/stop.sh >/dev/null 2>&1
args=""; [ $# -gt 0 ] && args="--ros-args $*"
nohup bash -c "source /root/dodge/env.sh && exec python3 /root/dodge/go2_dodge.py $args" > /root/dodge/dodge.log 2>&1 < /dev/null &
sleep 5; tail -n 2 /root/dodge/dodge.log
