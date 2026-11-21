# Durance Vision Lab @ Berlin HackLab Robotics — Track-Notizen

Quellen: [Durance_Vision_Lab_Track.pdf](Durance_Vision_Lab_Track.pdf), [riversdk/](riversdk/index.md) (Spiegel von https://riversdk.dev/docs/, Stand 2026-11-21).
Markiert mit **(Annahme)** ist alles, was nicht wörtlich in den Docs steht.

## Aufgabe

"The best demo based on RiverFPGA & Unitree Go2": Event-Kamera → SNN auf FPGA → ROS 2 → Roboterverhalten.

| Level | Inhalt |
| --- | --- |
| 01 Remix | Vorhandenen Gesten-Erkenner nutzen, **neue** ROS-2-Interaktion / zustandsbehaftetes Verhalten bauen. Kein umbenanntes Starter-Demo. |
| 02 Build (Flagship) | Eigene Daten aufnehmen + labeln, eigenes SNN mit riverops trainieren, per RiverSDK auf RiverFPGA deployen, ROS-2-Verhalten auf dem Go2. |
| 03 Boss mode | Level 2 robust machen (ungesehene Aufnahmen, Distraktoren, Target-Verlust/Reacquisition, Kamerabewegung) mit **messbarer** Verbesserung. |

Bewertung: Working System 35 · Robustness & control 25 (Wiederholbarkeit, unbekannte Inputs, Timeouts, sichere Defaults) · Originality 20 · Engineering choices 10 · Reproducibility 10.

Abgabe:
- 2-Minuten-Demo (live oder Video): Sensor-Input, On-Board-Inferenz, Roboter-Reaktion — oder deklarierter Hardware-Fallback. Klar trennen: neu vs. Starter-Demo.
- Repo mit README, Launch-/Trainingsschritten, Modell/Config, ROS-2-Code, Datensatz-Quellen + Splits, Contribution Summary. **Durance-Tools und -Datensatz nicht veröffentlichen** (auch `docs/riversdk/`, Wheels, SDK nicht ins öffentliche Repo).
- Zahlen: Erfolge/Versuche, False Triggers, gemessene Sensor-zu-Kommando-Latenz (getrennt von reiner Inferenzzeit), Testbedingungen, Fehlerfälle.

## Hardware

- Pro Team: Kria KV260 (RiverFPGA) + Prophesee GenX320 (320×320 Event-Sensor, MIPI).
- Geteilt: ein Go2-Workbench mit integriertem Kria + Kamera.
- Kria: RiverOS (Yocto), `ssh root@durance-kria.local` (kein Passwort) oder UART `minicom -D /dev/ttyUSB1 -b 115200`. eth0 = 192.168.77.1 (Laptop), USB-Ethernet-Adapter = Go2-Netz 192.168.123.0/24 (Go2 = .161, Kria = .99).

## Pipeline auf dem Kria

```
GenX320 --MIPI--> despot (Events -> Frames, 100 Hz, Downsampling) --> spleat-server/backward-server (Port 3333) --DMA--> RIVER IP (PL)
                                                                         |
                                                    ROS-2-Node (Humble, CycloneDDS) --Ethernet--> Go2
```

- `despot_test_camera` einmal nach jedem Reboot, dann `despot_send_and_forget --to_display 0`.
- `systemctl status spleat-server.service`; Config `/root/.config/spleat-server/config.yaml` (Input 2×80×80, Output 5×1×1, scaling 4096 = Q4.12).
- `/root/metavision_camera.json`: frequency 100 Hz, downsampling 4 (→ 80×80), noise filter, Biases.
- `ros2 launch bringup pipeline.launch.py despot:=false port:=3333` (Port = der in `config.yaml`; Docs-Beispiel nennt 41017, mit `ss -ltn` prüfen).

## Modell-Deploy (Level 2)

1. Training mit riverops (PyTorch, Input `[B, T, C, H, W]`, `SpikingConvBN2d`, Spike-Count-Readout).
2. `export_for_river(model.net, example)` → Modell-JSON; `prepare_qat` + `export_params()` → `weights.txt` (Weights/Bias Q1.7, Threshold/Leak Q4.12, keine Leerzeile am Ende).
3. Bitstream: riversdk.dev → "Open firmware jobs" → "New firmware" (Board KV260, IP GenX320) → `spleat.bit.bin` + `spleat.dtbo` nach `/usr/lib/firmware/xilinx/spleat/`.
4. `config.yaml`, `metavision_camera.json`, `weights.txt` per scp aktualisieren, ROS-Node anpassen.

Export-Grenzen: nur Teilmenge der Ops, keine Zyklen, kein Multi-Input/-Output, kein String-Padding, Linear-Head nicht exportierbar, Reset-Mode nicht im Graph.

## Auffälligkeiten / offene Fragen an Durance

- Notebook trainiert auf **32×32**, despot liefert 320/2^k (80, 40, 20 …) und die Default-Config sagt 80×80. Eingangsgröße fürs eigene Modell vorab klären. **(Annahme: auf 80×80 oder 40×40 trainieren.)**
- Wie lange dauert ein Firmware-Job? Reicht bei gleicher Architektur ein neues `weights.txt` ohne neuen Bitstream? **(Annahme: ja — wäre der schnellste Iterationspfad.)**
- Dateinamen weichen ab: Notebook erwartet `DuranceAI_motion_buddy_dataset.zip`, Download heißt `durance-ai-handgesture-recognition.zip`; Docs `sdk_x86_64_installer.sh` vs. Download `x86_64_sdk_installer.sh`.
- Port: config.yaml sagt 3333, Launch-Beispiel 41017.
- **Starter erwartet 11 Scores (DVS-Gesture), Hackathon-Modell/Config liefert 5** (siehe Starter-Code). Welches Modell läuft auf den ausgegebenen Boards?
- Darf der (angepasste) Starter-ROS-Code ins öffentliche Abgabe-Repo?
- Datensatz-Klassen: Sit, Lay down, Hello, Dance, No action; Sessions 01–03 Train, 04 Val, 05 Test.

## Starter-Code (`starter/go2-kria/`, aus go2-kria-generated.zip)

```
SPLEAT-Server (TCP 3333, CBOR) -> gesture_node -> /durance/gestures -> action_bridge -> /api/sport/request -> Go2
Go2 /utlidar/robot_pose + /utlidar/height_map_array -> obstacle_detection -> /durance/clearance -^
```

- `gesture_node`: TCP-Client zum SPLEAT-Server (`host`, `port`, `window` = 50). Liest CBOR-Frames mit 4-Byte-Big-Endian-Länge, nimmt `payload` nur bei **genau 11 float32 (44 Byte)**, gleitender Mittelwert, publiziert `msgs/Gestures` (`float32[11] scores`). Klassen in DVS-Gesture-Reihenfolge (0 Clap, 1 R wave, 2 L wave, 3 R arm cw, 4 R arm ccw, 5 L arm cw, 6 L arm ccw, 7 Arm roll, 8 Air drums, 9 Air guitar, 10 Other). Andere Payload-Größen werden verworfen → für ein 5-Klassen-Modell muss der Node (und `msgs/Gestures`) angepasst werden.
- `action_bridge`: argmax → Tabelle Klasse → Sport-`api_id` (1009 sit, 1004 stand up, 1005 lay down, 1010 rise from sit, 1016 hello, 1017 stretch, 1020 content, 1022 dance, 1029 greet, 1036 heart, Other = nichts). Feste Sperrzeit pro Aktion (3–15 s), nach sit/lay down automatisch STAND_UP. Antwort-Code ≠ 0 gibt sofort frei. **Sendet nur bei Clearance == CLEAR.**
- `obstacle_detection`: Lidar-Höhenkarte um die Roboterpose (`radius` 0.5, `blind_radius` 0.35, `min_height` 0.2) → `msgs/Clearance` (CLEAR / OBSTACLE / UNKNOWN). Ohne Go2-Lidar-Daten bleibt es UNKNOWN → keine Bewegung, auch nicht am Tisch ohne Roboter.
- `profiling_node` (`/profiling/fps_gestures`), `debug_node gesture_monitor` (curses-Anzeige der Scores).
- Mocks: `robot_sim` (quittiert Sport-Requests, keine Physik), `qualia_sim` (falscher SPLEAT-Server, zufällige 11er-Scores alle 2 s), `scripts/smoke.py` (End-to-End ohne Hardware, DDS-Domain 73).
- Build: `sdk/installer.sh` (x86_64-SDK) → `./manage.sh image` → `check` → `build` → `package`. Im Zip liegt schon ein fertiges `kria-runtime.tar.gz` (unveränderter Starter, direkt deploybar).
- Eigene Physik-Sim für das Verhalten: [../sim/mujoco/README.md](../sim/mujoco/README.md) (MuJoCo + Sport-API-Emulator auf DDS-Domain 1).
