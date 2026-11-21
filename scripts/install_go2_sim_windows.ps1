# Native-Windows runtime for the Go2 dodge sim and 3D viewer (the WSLg window does not show up on this machine).
# Needs the WSL install first (scripts/install_go2_sim.sh); copies the Go2 model and the policy from there.
#   powershell -ExecutionPolicy Bypass -File scripts\install_go2_sim_windows.ps1
$ErrorActionPreference = 'Stop'
$dest = Join-Path $HOME 'go2_sim'
$ck = 'walk-these-ways-go2/runs/gait-conditioned-agility/pretrain-go2/train/142238.667503/checkpoints'
$wslDest = '/mnt/c/Users/' + $env:USERNAME + '/go2_sim'
wsl -d Ubuntu-22.04 -u root -- bash -c "set -e; mkdir -p $wslDest/$ck $wslDest/unitree_mujoco/unitree_robots; cp /root/go2_sim/$ck/body_latest.jit /root/go2_sim/$ck/adaptation_module_latest.jit $wslDest/$ck/; cp -r /root/go2_sim/unitree_mujoco/unitree_robots/go2 $wslDest/unitree_mujoco/unitree_robots/"

$venv = Join-Path $dest 'venv-win'
if (-not (Test-Path $venv)) { py -3.12 -m venv $venv }
$py = Join-Path $venv 'Scripts\python.exe'
& $py -m pip install -q --upgrade pip
& $py -m pip install -q mujoco numpy pillow matplotlib trimesh fast-simplification   # last two: low-poly meshes for the viewer
& $py -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
& $py -c "import mujoco, torch; print('mujoco', mujoco.__version__, '| torch', torch.__version__)"
Write-Output "GO2_SIM_WINDOWS_READY: $dest"
