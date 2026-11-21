<#
One-time laptop setup for the direct cable to the Go2. Run it yourself in an ADMINISTRATOR PowerShell:

    Get-NetAdapter                                   # plug in the USB-Ethernet adapter, note its Name
    .\setup_laptop.ps1 -Adapter "Ethernet 3"          # the name from above
    .\setup_laptop.ps1 -Adapter "Ethernet 3" -Undo    # back to DHCP, rules removed

What it changes (nothing else):
  1. the wired adapter gets the fixed address 192.168.123.222/24, no gateway (internet stays on WLAN)
  2. WSL's Hyper-V firewall (mirrored networking, default inbound = Block) gets one rule:
     allow inbound UDP from 192.168.123.0/24 only (the robot's DDS traffic)
  3. the Windows firewall gets the same rule, limited to that adapter
#>
param(
    [Parameter(Mandatory = $true)][string]$Adapter,
    [string]$Ip = "192.168.123.222",
    [switch]$Undo
)
$ErrorActionPreference = "Stop"
$WslVm = '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}'
$RuleName = 'Go2-DDS-In'
$null = Get-NetAdapter -Name $Adapter          # fails early if the name is wrong

if ($Undo) {
    Remove-NetIPAddress -InterfaceAlias $Adapter -IPAddress $Ip -Confirm:$false -ErrorAction SilentlyContinue
    Set-NetIPInterface -InterfaceAlias $Adapter -Dhcp Enabled
    Remove-NetFirewallHyperVRule -Name $RuleName -ErrorAction SilentlyContinue
    Remove-NetFirewallRule -Name $RuleName -ErrorAction SilentlyContinue
    Write-Host "undone: $Adapter back on DHCP, $RuleName rules removed"
    return
}

# 1. fixed address on the Go2 network
Set-NetIPInterface -InterfaceAlias $Adapter -Dhcp Disabled
Get-NetIPAddress -InterfaceAlias $Adapter -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -ne $Ip } | Remove-NetIPAddress -Confirm:$false
if (-not (Get-NetIPAddress -InterfaceAlias $Adapter -IPAddress $Ip -ErrorAction SilentlyContinue)) {
    New-NetIPAddress -InterfaceAlias $Adapter -IPAddress $Ip -PrefixLength 24 | Out-Null
}

# 2. WSL (Hyper-V firewall)
if (-not (Get-NetFirewallHyperVRule -Name $RuleName -ErrorAction SilentlyContinue)) {
    New-NetFirewallHyperVRule -Name $RuleName -DisplayName "Go2 DDS into WSL (UDP from 192.168.123.0/24)" `
        -Direction Inbound -VMCreatorId $WslVm -Protocol UDP -RemoteAddresses 192.168.123.0/24 -Action Allow | Out-Null
}

# 3. Windows firewall
if (-not (Get-NetFirewallRule -Name $RuleName -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -Name $RuleName -DisplayName "Go2 DDS (UDP from 192.168.123.0/24)" -Direction Inbound `
        -Protocol UDP -RemoteAddress 192.168.123.0/24 -InterfaceAlias $Adapter -Action Allow | Out-Null
}

Write-Host "done: $Adapter = $Ip/24, firewall rules $RuleName added. Restart WSL once: wsl --shutdown"
Write-Host "robot reachable?"
Test-Connection 192.168.123.161 -Count 2 -ErrorAction SilentlyContinue | Format-Table -AutoSize
