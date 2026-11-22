#!/bin/bash
# Stop go2_dodge.py on the Kria (its Ctrl+C handler sends StopMove).
for p in $(ps | grep "[g]o2_dodge.py" | grep python3 | awk '{print $1}'); do kill -INT "$p"; done
sleep 1; ps | grep -q "[g]o2_dodge.py" && echo "still running" || echo "stopped"
