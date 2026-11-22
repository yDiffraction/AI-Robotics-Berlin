# Go2-Simulation (MuJoCo)

Physik-Sim des Unitree Go2 über [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) (Python-Variante).
Unabhängig von der Gazebo-/Docker-Sim in `sim/` (`sim.sh`, `ros_ws/`): läuft ohne ROS direkt auf dem Host.

Einmalig einrichten (beide Klone sind per `.gitignore` aus dem Repo ausgenommen):

```bash
git clone https://github.com/unitreerobotics/unitree_mujoco.git third_party/unitree_mujoco          # getestet mit 1eb6642
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git third_party/unitree_sdk2_python  # getestet mit 814556d
conda create -n durance python=3.10 && conda run -n durance pip install mujoco pygame cyclonedds==0.10.2 -e third_party/unitree_sdk2_python
```

In `third_party/unitree_mujoco/simulate_python/config.py` `USE_JOYSTICK = 0` setzen, damit kein Gamepad nötig ist;
`DOMAIN_ID = 1` und `INTERFACE = "lo"` sind dort schon Standard.

Starten (aus dem Repo-Wurzelverzeichnis):

```bash
# Terminal 1: Simulator (MuJoCo-Viewer)
cd third_party/unitree_mujoco/simulate_python && conda run -n durance --no-capture-output python unitree_mujoco.py

# Terminal 2: Sport-Emulator, Go2 steht auf und wartet auf Requests
conda run -n durance --no-capture-output python sim/mujoco/sport_emulator.py

# Terminal 3: einzelne Aktion schicken, so wie es action_bridge tut (Name oder api_id)
conda run -n durance --no-capture-output python sim/mujoco/send_action.py hello
```

`check_sim.py` ist der Low-Level-Smoke-Test ohne Emulator (nicht gleichzeitig mit dem Emulator starten, beide schreiben `rt/lowcmd`).

- Sim läuft auf DDS-Domain **1**, Interface `lo` (`simulate_python/config.py`). Der echte Go2 / Kria nutzt Domain 0, so kann ein Sim-Kommando nie den echten Roboter erreichen.
- unitree_mujoco kennt nur Low-Level: `rt/lowcmd` rein, `rt/lowstate` + `rt/sportmodestate` raus. Den High-Level-Sport-Service des echten Go2 ersetzt `sport_emulator.py`: liest `rt/api/sport/request` (ROS-Topic `/api/sport/request`, das `action_bridge` aus dem Starter publiziert), antwortet auf `rt/api/sport/response` (Code 0, bei unbekannter api_id 3203) und spielt die passende Keyframe-Bewegung aus `sport_actions.py` ab.
- Emulierte api_ids: 1001 damp, 1002 balance stand, 1004 stand up, 1005 stand down, 1006 recovery stand, 1009 sit, 1010 rise from sit, 1016 hello, 1017 stretch, 1020 content, 1022 dance, 1029 und 1036 (beide als hello angenähert). Ein neuer Request bricht die laufende Bewegung ab.
- Die Bewegungen sind handgetunte PD-Keyframes, nicht Unitrees Controller: gut, um Verhaltenslogik zu prüfen (welche Aktion wann, Reihenfolge, Abbruch), nicht um echte Dynamik oder Dauer zu beurteilen. Kein Laufen (`Move`), kein Lidar (`/utlidar/*`).
