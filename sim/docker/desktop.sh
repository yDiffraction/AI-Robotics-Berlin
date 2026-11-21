#!/bin/bash
# Start a virtual X display and serve it to the browser via noVNC on port 6080.
set -e
mkdir -p ~/.vnc
vncserver :1 -geometry 1600x900 -depth 24 -SecurityTypes None -localhost no \
    -xstartup /usr/bin/fluxbox >/tmp/vnc.log 2>&1 || true
websockify --web /usr/share/novnc 6080 localhost:5901 >/tmp/novnc.log 2>&1 &
echo "Desktop ready: http://localhost:6080/vnc.html"
