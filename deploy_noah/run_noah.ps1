<#
Noah's throw reflex (his ROS 2 nodes) on the real Go2, from a normal PowerShell window. Runs in WSL Ubuntu-22.04
with ROS 2 Humble; the workspace lives in /root/noah_ws. Nothing here touches sim/deploy.

    .\run_noah.ps1 build           # once (and after pulling Noah's changes): build /root/noah_ws
    .\run_noah.ps1 check           # wired interface, ping dog (.161) and Kria (.99), SPLEAT port 3333
    .\run_noah.ps1 topics          # do LiDAR height map / pose arrive from the dog? (sends nothing)
    .\run_noah.ps1 desk            # no hardware: emulator + fake LiDAR + robot_sim, loopback only
    .\run_noah.ps1 dry             # real Kria + real LiDAR, commands go to /dry/api/sport/request only
    .\run_noah.ps1 robot           # the real dog
    .\run_noah.ps1 robot dodge_speed:=0.3      # extra arguments are ros2 launch arguments
#>
param(
    [ValidateSet("build", "check", "topics", "desk", "dry", "robot", "sdk")][string]$What = "check",
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest = @()
)
$here = "/mnt/" + $PSScriptRoot.Substring(0, 1).ToLower() + $PSScriptRoot.Substring(2).Replace('\', '/')
if ($What -eq "build") {
    wsl -d Ubuntu-22.04 -u root -- bash "$here/real/build_ws.sh" @Rest
} else {
    wsl -d Ubuntu-22.04 -u root -- bash "$here/real/run.sh" $What @Rest
}
