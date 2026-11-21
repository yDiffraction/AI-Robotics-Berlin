"""Make the Go2 CHAMP Gazebo launch load its controllers reliably.

Upstream (anujjain-dev/unitree-go2-ros2, champ_gazebo/launch/gazebo.launch.py) loads
controllers with `ros2 control load_controller` at the same moment Gazebo starts.
That CLI goes through the ros2 daemon; if the daemon is stale or the controller
manager is not up yet, loading fails and the robot collapses on spawn.

This swaps them for controller_manager's `spawner`, which talks to the controller
manager directly and waits for it. Idempotent.
"""
import re
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else
            "/root/go2_ws/src/unitree-go2-ros2/champ/champ_gazebo/launch/gazebo.launch.py")
src = path.read_text()

if "executable=\"spawner\"" in src:
    print(f"already patched: {path}")
    sys.exit(0)

pattern = re.compile(
    r"(?P<var>\w+) = ExecuteProcess\(\s*cmd=\['ros2', 'control', 'load_controller', "
    r"'--set-state', 'active',\s*'(?P<ctrl>\w+)'\],\s*output='screen',?\s*\)",
    re.MULTILINE,
)


def repl(m):
    return (
        f"{m['var']} = Node(\n"
        f"        package=\"controller_manager\",\n"
        f"        executable=\"spawner\",\n"
        f"        arguments=[\"{m['ctrl']}\", \"--controller-manager-timeout\", \"120\"],\n"
        f"        output=\"screen\",\n"
        f"    )"
    )


new, n = pattern.subn(repl, src)
if n == 0:
    sys.exit(f"pattern not found in {path}; upstream file changed?")
path.write_text(new)
print(f"patched {n} controller loaders in {path}")
