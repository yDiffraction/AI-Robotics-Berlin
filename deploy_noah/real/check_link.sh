#!/usr/bin/env bash
# Network check for the wired laptop -> Go2 (+ Kria on the dog). Read-only.
if ! ip -4 -br addr | grep 192.168.123; then
    echo "NO interface with 192.168.123.x: run sim/deploy/setup_laptop.ps1 (admin), then wsl --shutdown"
fi
echo "--- dog 192.168.123.161"
ping -c 2 -W 1 192.168.123.161 | tail -2
echo "--- Kria 192.168.123.99"
ping -c 2 -W 1 192.168.123.99 | tail -2
if timeout 2 bash -c "</dev/tcp/192.168.123.99/${1:-3333}" 2>/dev/null; then
    echo "SPLEAT port ${1:-3333} open"
else
    echo "SPLEAT port ${1:-3333} NOT reachable (Kria up? spleat-server running?)"
fi
