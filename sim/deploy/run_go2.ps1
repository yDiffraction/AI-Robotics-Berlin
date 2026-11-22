<#
Start the Go2 tools in WSL (Ubuntu-22.04, /root/go2_sim/venv) from a normal PowerShell window.

    .\run_go2.ps1 check     # network: wired interface, ping the robot
    .\run_go2.ps1 probe     # read-only: robot state rate, remote buttons, remote topic (safe in sport mode)
    .\run_go2.ps1 gentle    # same as below, but the dodge at reduced sideways speed (use this first)
    .\run_go2.ps1           # go2_modes.py: sport mode <-> our controller, STILL dodge on A / D-pad
    .\run_go2.ps1 trot      # the previous SAFE dodge (trots ~2 s in place before the kick)
#>
param([ValidateSet("modes", "gentle", "trot", "probe", "check")][string]$What = "modes")
$deploy = "/mnt/" + $PSScriptRoot.Substring(0, 1).ToLower() + $PSScriptRoot.Substring(2).Replace('\', '/')
$py = "/root/go2_sim/venv/bin/python"
switch ($What) {
    "check" {
        wsl -d Ubuntu-22.04 -u root -- bash -c "ip -4 -br addr | grep 192.168.123 || echo 'NO interface with 192.168.123.x (setup_laptop.ps1, then wsl --shutdown)'; ping -c 3 -W 1 192.168.123.161"
    }
    "probe" { wsl -d Ubuntu-22.04 -u root -- $py -u "$deploy/remote_probe.py" --iface auto }
    "modes" { wsl -d Ubuntu-22.04 -u root -- $py -u "$deploy/go2_modes.py" }
    "gentle" { wsl -d Ubuntu-22.04 -u root -- $py -u "$deploy/go2_modes.py" --gentle }
    "trot" { wsl -d Ubuntu-22.04 -u root -- $py -u "$deploy/go2_modes.py" --trot-start }
}
