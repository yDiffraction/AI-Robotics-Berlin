# Open the side-by-side Go2 dodge viewer as a native Windows window.
#   powershell -ExecutionPolicy Bypass -File scripts\show_dodges.ps1            # half speed
#   powershell -ExecutionPolicy Bypass -File scripts\show_dodges.ps1 -Speed 0.25
param([double]$Speed = 0.5)
$py = Join-Path $HOME 'go2_sim\venv-win\Scripts\python.exe'
if (-not (Test-Path $py)) { throw "Run scripts\install_go2_sim_windows.ps1 first." }
Push-Location (Join-Path $PSScriptRoot '..\sim')
try { & $py -W ignore show_dodges.py --speed $Speed } finally { Pop-Location }
