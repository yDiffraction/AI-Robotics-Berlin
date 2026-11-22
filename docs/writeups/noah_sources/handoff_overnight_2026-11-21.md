# Handoff: Ballwurf-Trigger (Stand 2026-11-21, 23:50)

Ziel: Der Go2 weicht einem im hohen Bogen geworfenen Ball aus. Er reagiert erst **nach dem Abwurf** (Timing manuell getestet), Täuschungen und Ruhe sollen nicht auslösen.
Jury: Working System 35 · Robustness & control 25 · Originality 20 · Engineering 10 · Reproducibility 10.

Alles ist gestoppt: GPU-Warteschlange, drei Hintergrund-Agenten, Wächter. Auf der Platine wurde **nichts** verändert. Auf dem Kria laufen weiter Tims Gewichte (md5 9ff58e0e…, gleich `runs/tim_fpga/export/weights.txt`) und `despot_debug_camera` → Noahs `record_tool.py`.

## 1. Kernbefunde (wichtigste zuerst)

1. **Die Information steckt im Luftraum über dem Werfer, nicht in der Armbewegung.**
   - Stolperdraht = aktive Zellen in den oberen 8 Zeilen (80×80), summiert über 2 Bilder. In 02/03 erkennt er **42/42 Würfe** bei **0/65 Täuschungen**, etwa 90 ms nach dem Abwurf. Auf Ruhe in der Wurf-Szene gibt es 0,6 Fehlauslöser pro Minute.
   - Grenzen: in der unruhigen Ruhe-Szene 04/211821 23/min, bei 04-Täuschungen 27/36 (Kopf am oberen Rand).
   - Das Vollbild-SNN schafft nur AUC ≈ 0,53–0,59 für Wurf gegen Täuschung.
   - Alle Werte sind in-sample, nicht kreuzvalidiert. Prototyp: `/tmp/claude-1000/-home-noah--claude-spares-spare-QrzDK9/50630430-7638-433c-b696-13b6ccca033d/scratchpad/roi_trigger.py`. Der Scratchpad ist flüchtig, bei Bedarf ins Repo kopieren.
2. **Kausaler klassischer Tracker** (`runs/ball/tracker/causal_tracker.py`, Variante `default`):
   - Er findet in 02/03 34/42 Würfe, davon 32/33 der getrackten innerhalb von 100 ms.
   - Verzögerung im Median +4 Bilder (40 ms) nach dem Abwurf.
   - Täuschungen **0/104 über alle Datensätze, auch 04**.
   - Ruhe: unruhige Szene 0,2/min, Wurf-Szene 0,8/min.
   - Die Parameter wurden auf diesen Daten getuned, also optimistisch. Ablationen stehen in `runs/ball/tracker/eval_final.txt`. Besonders wichtig sind der Konstant-Geschwindigkeits-Fit, die Residuen-Grenze und das Isolationskriterium.
3. **Tims bisheriges SNN-Setup (b0_tim) ist schwach.** Kreuzvalidiert, Folds A und B (C wurde abgebrochen):
   - AUC Wurf gegen Täuschung 0,59 und 0,53.
   - Bei 10 % Fehlauslösern auf Täuschungen erkennt es nur 19 % bzw. 6 % der Würfe.
   - Ein eingesetzter Ball löst in 22–30 % der Fälle aus. Das Modell nutzt den Ball also kaum.
   - Tims `runs_final/final_tso_sym.pt` nennt laut Panel **alle** Würfe und Täuschungen „Ruhe“. Nicht deployen.
4. **Fehler in Tims Trainingscode** (`~/robust/robust_cv.py` auf dem A100; lokale Kopie `runs/robust_tim/`):
   - Beim zeitlichen Verschieben (±8 Bilder) und Strecken der Fenster bleibt das Ziel bei Frame 70. Es liegt dadurch um bis zu ±80 ms falsch, bei 02 bis zu 200 ms vor dem Abwurf.
   - `prep_raw.prep_clips` ignoriert `release_ms`.
   - `counts()` zählt ab dem ersten Event statt ab dem Clip-Start. Ruhe-Clips sind dadurch um bis zu 222 ms verschoben.
   - `idle_idx` spielt Ruhe-Fenster im Ping-Pong teils rückwärts ab. Diesen Effekt gibt es nur im Training.
   - `update_bn` läuft auf augmentierten, also ausgedünnten Daten.
   - Die `ts`-Loss (kumulatives Mittel ab dem Fensterstart) verwässert den Ball-Impuls. `--sym` erzwingt Gleichheit genau in den Bildern, in denen der Ball in 03 sichtbar ist (61–69).
   - Eine vertikale Verschiebung um 8 Zellen schiebt den Ball aus dem Bild.
   - Alles ist in `runs/ball/ball_cv.py` behoben. Tim sollte es in `robust_cv.py` übernehmen.
5. **Daten:**
   - In 04/215305 (28 Würfe, 36 Täuschungen) ist der Ball nach dem Abwurf **nicht sichtbar**, weil der Kopf am oberen Rand ist. In 01 ist die Person zu nah.
   - Die Wurf-Labels bedeuten dort nur „Armbewegung“, was Täuschungen gezielt nachahmen.
   - Alle Ruhe-Clips im neuen Datensatz stammen aus einer völlig anderen, unruhigen Szene (Decke, Bewegung im ganzen Bild).
   - Würfe und Täuschungen haben eine ruhige Kamera und eine Person. Daraus entsteht die Abkürzung „volles Bild = Ruhe“.
   - Die Aufnahmen 201452, 201622 und 201703 enthalten nur Würfe.
6. **Deploy-Risiken (Working System):**
   - despot schickt live **Zählwerte**, trainiert wird auf **0/1**. Das alte 5-Klassen-Modell ist damit auf der Platine kollabiert (`ref5_count` gegen `ref5_binary`).
   - Live kamen nur etwa 47 statt 100 Ergebnisse/s an, in Schüben (`live1.npz`).
   - `gesture_trigger.py` setzt 3 Klassen voraus: mit 2 Ausgängen gibt es einen IndexError, und die Verhältnisregel löst auf einzelne Spikes aus. `gesture_node` mittelt über 50 Bilder und verwässert den Ball-Impuls von 5–10 Bildern.
   - Ein Deploy nur durch Tausch der Gewichte braucht die v1-Architektur mit **genau 11 Ausgängen**. Ein Modell mit 2 Ausgängen muss aufgefüllt werden. `export_tim.py` bricht bei `arch`-Checkpoints.
   - Die Fixpunkt-Darstellung (Q2.6/Q4.12) bringt laut Panel bei feinjustierten Modellen +18 % Output-Spikes. Schwellen deshalb auf der Fixpunkt-Simulation kalibrieren, nicht auf Float.
7. **Recherche:** Alle Systeme, die tatsächlich Bällen ausweichen oder sie fangen, erkennen den **Ball** direkt und prüfen die Physik: Falanga-Drohne 2020, ANYmal (Forrai 2023), Mini-Cheetah (Zhu 2023), Delbrucks Torwart, EV-Catcher. EV-Catcher nutzt binäre Bilder. Vorarbeit zur Erkennung des Abwurf-Moments mit Event-Kamera wurde **nicht gefunden**, das ist ein Originalitäts-Argument. Laut REBot (Go2) braucht das Ausweichen mehr als 0,5 s Vorwarnung, also sollte der Trigger innerhalb von etwa 100 ms kommen.

## 2. Was gebaut wurde (Dateien)

| Datei | Zweck | Stand |
|---|---|---|
| `tools/mark_release.py` | Abwurf-Zeitpunkt per Tracker → Spalte `release_ms` in `labels.csv` (33/77 Würfe aus 02/03, 610–870 ms) | fertig, Tim hat das Format abgesegnet |
| `dataset/_dataset01_01020304/labels.csv`, `preview_release.png` | `release_ms` eingetragen, Kontrollbild | fertig, nicht committet |
| `training/prep_clips.py`, `tools/build_dataset.py` | lesen bzw. kennen `release_ms` (leer = 700 ms) | fertig |
| `runs/ball/make_bank.py` | Clip-Bank: Fenster am 700-ms-Anker, `release` getrennt, Clip-Zeitachse, `--extra-idle` (+255 Ruhe-Clips aus der Wurf-Szene aus `_dataset01_010203`) → `bank_mix.pt` (724 Clips) | fertig |
| `runs/ball/ball_tracks.py` → `tracks.npz` | Ball-Zellen pro Bild für die 33 getrackten Würfe (für Einsetzen, Entfernen, Hilfskopf) | fertig |
| `runs/ball/ball_cv.py` | Vergleichs-Pipeline auf Basis von Tims Code, alle Bugs aus Abschnitt 1.4 behoben | fertig, CPU-Smoke-Tests ok |
| `runs/ball/compare.py` | Tabelle und gepaarter Bootstrap; Hauptkennzahl `prim` = AUC Wurf gegen Täuschung innerhalb der Folds, nur 02/03 | fertig |
| `runs/ball/queue.sh`, `jobs.txt` | GPU-Warteschlange (gestoppt, `STOP`-Datei liegt) | siehe Abschnitt 3 |
| `runs/ball/tracker/` | kausaler Tracker mit Auswertung (`eval_final.txt`, `per_clip.csv`) | Agent abgebrochen, Auswertung liegt vor, `RESULTS.md` fehlt |
| `training/export_ball.py` | Export eines ball_cv-Checkpoints → `weights.txt` (11 Ausgänge), `config.yaml`, Fixpunkt-Check | **halbfertig, ungetestet** (Agent abgebrochen) |
| `tools/spleat_binarize_proxy.py` | Weiterleiter auf dem Kria: despot → (Zählwerte→0/1) → spleat-server :3334 | **halbfertig, ungetestet** |
| `tools/ball_trigger_logic.py` (+ Kopie `sim/ros_ws/src/go2_sim_tools/go2_sim_tools/ball_trigger_logic.py`), `tests/test_ball_trigger_logic.py` | Auslöse-Logik: Fenster, Schwelle, Vorbereiten/Ausweichen, Refraktärzeit, Watchdog | **halbfertig**; ROS-Knoten `ball_trigger.py` fehlt |

`ball_cv.py`-Optionen (alle per Flag):
- `--loss win|mil|ts`
- `--synth-throw` + `--paste-idle 0` (Ball nur in Täuschungen einsetzen), `--erase-share`
- `--aux-ball` (Hilfskopf für die Ball-Position, nur im Training), `--pop M`
- `--swing` (3. Ausgang „Ausholen“ für den zweistufigen Reflex; Gate-Kennzahlen `*_gated`)
- `--roi-rows Y` (Netz sieht nur den Luftraum), `--drop-throws 04,01`, `--idle-train busy|scene|both`
- `--legacy-rel` (Tims Bug nachbilden), `--final`

Folds: A = 200828 + 201703, B = 200001 + 195808 + 201452, C = 170441 + 201622 + d01, D = 215305 (Ball unsichtbar, nur zum Testen). Ein Lauf dauert auf der RTX 4070 etwa 25 min (3 Folds × 8 min). Nie zwei Läufe gleichzeitig: 7,4 von 8 GB sind belegt.

## 3. Gestoppter Zustand der Experimente

- Fertig: nur `b0_tim`, Folds A und B (Zahlen in Abschnitt 1.3; Log `runs/ball/logs/b0_tim.log`, Modelle `runs/ball/runs_ball/b0_tim/fold_*.pt`).
- **Nicht gelaufen** (Reihenfolge in `runs/ball/jobs.txt`):
  - w1_win (Referenz)
  - final_w1
  - **r1_roi16, r2_roi24 (Luftraum-SNN, meiner Meinung nach am wichtigsten)**
  - d1_drop, w1_win_s1 (zweiter Seed als Rauschgrenze)
  - w2_paste, a1_aux, m1_mil, w3_sceneidle, t1_swing
  - b1_fix, p1_pop5, w4_erase
- Neu starten:
  ```bash
  cd runs/ball && rm -f STOP done.txt && setsid nohup ./queue.sh > queue.out 2>&1 < /dev/null &
  ```
  Vorher die gewünschte Reihenfolge in `jobs.txt` setzen. `b0_tim` aus `jobs.txt` entfernen oder in `done.txt` eintragen.
- Auswerten:
  ```bash
  cd runs/ball && python3 compare.py --base w1_win
  ```

## 4. Was jetzt zu tun ist (nach Punkten priorisiert)

1. **Demo-Entscheidung treffen (Working System + Robustness).** Der Ball muss im Bild sichtbar sein: Kamera **statisch auf Stativ**, nicht auf dem Go2 (Eigenbewegung), Werfer etwa 3 m entfernt, Kopf ab Zeile 25–35 von 80, darüber freie Wand oder Decke (Aufbau wie 02/03, **nicht** wie 04).
2. **Trigger wählen.** Pragmatisch und am robustesten: Luftraum-SNN (`--roi-rows 16/24`) auf dem FPGA, optional per UND mit dem kausalen Tracker auf dem Host. Zuerst die beiden ROI-Läufe rechnen und mit `w1_win` vergleichen (`compare.py`). Wenn das SNN nicht an den Stolperdraht oder Tracker herankommt: Fallback ist der Tracker oder Stolperdraht auf dem Host als Bestätigung des SNN (wahrheitsgemäß so deklarieren).
3. **Eingabepfad angleichen.** Entweder:
   - (a) Live über den Laptop: `despot_debug_camera` → Laptop bildet binäre 10-ms-Bilder (wie bei den Aufnahmen) → SPLEAT-Server auf dem FPGA. Prüfen, ob `tools/live_vis.py` das schon kann. Damit entfallen Zählwerte, Rauschfilter und die Takt-Probleme.
   - (b) `tools/spleat_binarize_proxy.py` auf dem Kria fertigstellen und testen, dazu den Server-Port in `config.yaml` auf 3334 setzen.
4. **Finales Modell → Export → Board-Test.**
   - Training mit `ball_cv.py --final <beste Konfiguration>`.
   - Export mit `training/export_ball.py` fertigstellen (Kopf auf 11 Ausgänge auffüllen, 78501 Werte; Regressionstest gegen Tims Export).
   - Fixpunkt-Check auf Test-Clips.
   - Auf dem Board: Backup von `/root/weights.txt`, `scp` der neuen Gewichte, `systemctl restart spleat-server`, Test-Clips mit `runs/tim_fpga/replay.py --mode binary` abspielen und mit `compare_fpga.py` gegen die Fixpunkt-Simulation vergleichen. Dabei die Modell-Latenz aus dem Feld `latency` messen (Tims v1 liegt bei etwa 1,1 ms).
5. **Host-Logik.** `tools/ball_trigger_logic.py` fertigstellen (Tests in `tests/`), ROS-2-Knoten `ball_trigger.py` schreiben (TCP direkt zum SPLEAT-Server oder `/durance/gestures` mit `window=1`) → `/reflex/arm|dash|disarm` → `two_stage_reflex`. Den Entry-Point in `setup.py` von Hand eintragen. Die alte `gesture_trigger.py` nicht mit 2 Ausgängen benutzen.
6. **Kalibrierung vor Ort (Robustness, etwa 15 min).** Mit `record_tool.py` im Demo-Aufbau aufnehmen: 3 min Ruhe (Leute laufen, winken, heben den Ball auf), 20 Würfe, 20 Täuschungen. Darauf die Schwelle einstellen: höchstens 0,5 Ruhe-Fehlauslöser pro Minute und höchstens 5 % Täuschungen. Diese Zahlen sind die „unbekannten Bedingungen“ für die Jury.
7. **Zahlen für die Jury:**
   - Erfolge pro Versuch, Fehlauslöser (Täuschungen, Ruhe pro Minute).
   - **Latenz Sensor→Kommando** (240-fps-Handyvideo oder eine LED im Kamerabild, getrennt von der Modell-Latenz).
   - Fehlerfälle (Aufbau wie 04, Person zu nah).
   - Verbesserung gegenüber b0 (Level 3: messbar).
   - Folie zum Ball-Test: Ball entfernen → kein Auslösen, Ball in Täuschung einsetzen → Auslösen.
8. **Zweistufiger Reflex für die Demo (Control + Originality):** Beim Ausholen bereitmachen, beim Ball ausweichen, bei einer Täuschung sichtbar entspannen. Der Swing-Ausgang (`--swing`) erkennt Ausholen gegen Ruhe schon untrainiert mit AUC ≈ 0,89. Refraktärzeit 2 s, Watchdog (keine Bilder → kein Ausweichen) als sichere Defaults.
9. **Reproducibility:** `ball_cv.py`, `make_bank.py`, `ball_tracks.py`, `compare.py` und den Tracker aus `runs/` (git-ignoriert) nach `training/` bzw. `tools/` übernehmen, README ergänzen und committen. Durance-Material und `*.bin` außerhalb von `dataset/` nicht ins öffentliche Repo.

## 5. Quellen der Analysen (vollständige Texte)

- Panel (5 Blickwinkel + Plan): `~/.claude/projects/-home-noah-git-hackathons-hacklab-berlin-duranceai-robotics/50630430-7638-433c-b696-13b6ccca033d/subagents/workflows/wf_3c6f3c69-17a/journal.jsonl`
- Web-Recherche (4 Blickwinkel + Versuchszuordnung + Demo-Ideen): `…/wf_751c3e9e-849/journal.jsonl`
- Tims Läufe auf dem A100: `team-1-user@<A100>:~/robust/` (`runs_robust/summary.json`). Tim will `robust_cv.py`, `prep_raw.py` und `snn_model.py` auf seinen Branch committen und `release_ms` in `prep_raw.py` einlesen.
