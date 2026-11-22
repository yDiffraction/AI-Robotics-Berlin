# Write-up: Ball-Ausweich-Trigger (Noahs Agent-Sessions, 21.–22.11.2026)

Autor: Claude-Agent-Sessions an Noahs Laptop (Thread „Ultracode Handoff Status Review“ plus die Opus-„ultracode“-Session davor).
Stand: 22.11.2026, ca. 10:45. Alle Zahlen stammen aus Läufen in diesem Repo; Pfade sind relativ zum Repo-Root.
Rohmaterial (Panel, Web-Recherche, Nacht-Handoff, Ergebnis-JSONs): [`docs/writeups/noah_sources/`](noah_sources/).

> **Für die Zusammenfassung:** Abschnitt 1 ist die Kurzfassung. Abschnitt 11 listet die offenen Punkte, Abschnitt 12 die Dateien und Befehle.
> Was ist **gemessen** und was ist **Annahme**? Gemessen ist alles mit Zahl und Quelle. Annahmen sind als solche markiert.

---

## 1. Kurzfassung

1. **Ziel bis heute Morgen:** Der Go2 soll einem geworfenen Ball ausweichen, aber nicht bei Täuschungen (Wurfbewegung ohne Ball) und nicht in Ruhe.
   Event-Kamera → SNN auf dem Kria-FPGA → Host-Trigger → ROS 2 → Go2.
2. **Zentrale Erkenntnis:** Täuschungen unterscheiden sich von Würfen nur durch den **Ball im Luftraum über dem Werfer**, nicht durch die Armbewegung.
   - Tims ursprüngliches Setup (`b0_tim`) lag bei Wurf vs. Täuschung bei AUC 0,53–0,59, also Zufallsniveau.
   - Ein Großteil davon waren **Bugs im Trainingscode** (das Ziel lag zeitlich falsch, bis zu 200 ms vor dem Abwurf).
3. **Bestes Ball-Modell `w2_paste`:** Tims v1-Architektur, volles Bild. Im Training werden echte Ballflugbahnen in Täuschungs-Clips eingefügt und als „Wurf“ gelabelt.
   - Within-fold-AUC Wurf vs. Täuschung **0,991**, gepoolte AUC 0,965.
   - Ball-Test: Entfernt man den Ball, löst es nie aus (0 %). Fügt man ihn in Täuschungen ein, löst es in 76 % der Fälle aus.
   - Das Netz entscheidet also wirklich am Ball.
4. **Auf dem echten FPGA, ganze zurückgehaltene Aufnahmen (keine Clips), θ=5:**
   - 42/42 Würfe erkannt;
   - 8/65 Täuschungen lösen aus;
   - 1,7 Fehlauslöser pro Minute in den Pausen.
   - Kombiniert mit einem klassischen Ball-Tracker per UND: 34/42 Würfe, **0/65 Täuschungen, 0,1/min**.
5. **Eingabepfad gelöst:** Der Laptop baut selbst 0/1-Bilder aus den Kamera-Events und schickt sie an den SPLEAT-Server.
   - So schafft der Pfad 100 Bilder/s. despot dagegen schickt Zählwerte und kam live nur auf 47/s.
   - Modell-Latenz auf dem Board 0,45–0,6 ms, Round-Trip etwa 2 ms.
   - FPGA und Fixpunkt-Simulation stimmen überein (Spikes ±2 %, Entscheidungen pro Clip 95–100 %).
6. **Deployt und live gezeigt:** `final_w2_paste` lief auf dem Board, mit Live-Anzeige in der Kamera-Webseite (`tools/snn_live.py` in `record_tool.py`). Noah: „das Modell ist actually not bad“.
7. **Neue Aufgabe (seit 22.11., ca. 10:20):** Label-Wechsel. Wurf und Täuschung sind jetzt **dasselbe Label**: „Eine Wurfbewegung findet statt“, und das so früh wie möglich.
   - Tim hat dafür `~/early/early_cv.py` auf dem A100: ein Ausgang, EMA-Readout, Ziel ab 300–500 ms vor dem Abwurf.
   - Auf Tims zusammengesetzten Streams erkennt es 52–56 % der Würfe bei 1,5–2 Fehlalarmen pro Minute.
   - Auf **ganzen echten Aufnahmen** ist es deutlich schlechter: bei θ=0,95 nur 8–17 von 42 Würfen, und in den Pausen viele Fehlalarme.
   - Vermutete Ursachen und Verbesserungsvorschläge stehen in Abschnitt 9.

---

## 2. Aufgabe, Bewertung, Hardware (aus Track-Unterlagen)

Quelle: `docs/TRACK_NOTES.md`, `docs/Durance_Vision_Lab_Track.pdf`.

**Jury-Gewichtung:**

| Kriterium | Gewicht |
|---|---|
| Working System | 35 |
| Robustness & control | 25 |
| Originality | 20 |
| Engineering | 10 |
| Reproducibility | 10 |

**Levels:**
- **Level 2:** eigene Daten aufnehmen, eigenes SNN trainieren, auf das FPGA bringen, ROS-Verhalten bauen.
- **Level 3:** robust gegen ungesehene Aufnahmen und Distraktoren, mit **messbarer** Verbesserung.

**Abgabe:**
- 2-Minuten-Demo.
- Zahlen: Erfolge/Versuche, False Triggers und gemessene Latenz vom Sensor bis zum Kommando, getrennt von der reinen Inferenzzeit.
- Repo ohne Durance-Material.

**Hardware:**
- Prophesee GenX320 (320×320) per MIPI am Kria KV260 (RiverFPGA).
- Auf dem Kria: `despot` (Events → Frames, 100 Hz, Downsampling 4 → 80×80) → `spleat-server`/`backward-server` (TCP 3333, CBOR) → RIVER IP.
- Laptop über LAN: 192.168.77.1 = Kria, Laptop .198. Go2-Netz 192.168.123.0/24.

**Firmware-Randbedingung:**
- Auf dem Board läuft die arch-v1-Firmware mit **genau 11 Ausgängen** (Tims DVS-Gesture-Modell).
- Ein neues Modell lässt sich **nur durch Tausch der `weights.txt`** deployen, wenn der Graph identisch ist: 4 Spiking-Conv-Schichten 16/32/64/64, Stride 2, Kernel 5/3/3/3, Conv-Kopf, 11 Ausgänge, 78501 Werte.
- Fixpunkt: Gewichte und Bias Q2.6, Potentiale, Schwelle und Leak Q4.12.

---

## 3. Daten

| Satz | Inhalt | Brauchbarkeit für „Ball sichtbar“ |
|---|---|---|
| 01 | wenige Würfe und Täuschungen, Person zu nah | Ball trennt sich nicht vom Arm |
| 02 (`170441`) | 5 Würfe, 10 Täuschungen, ruhige Szene, zweite Person läuft vorbei | gut |
| 03 (`195808`, `200001`, `200828`, `201452`, `201622`, `201703`) | 37 Würfe, 55 Täuschungen | gut; 201452, 201622 und 201703 enthalten nur Würfe |
| 04 `215305` | 28 Würfe, 36 Täuschungen | **Ball nach dem Abwurf nicht sichtbar** (Kopf am oberen Rand): „Wurf“ heißt hier nur Armbewegung |
| 04 `211821` | lange unruhige Ruhe-Szene (Decke, Bewegung im ganzen Bild) | Quelle der „busy idle“-Clips |

- **Labels:** `dataset/0*/samples/labels.csv` enthält pro Aufnahme `label` (throw, feint, idle, invalid) und `decision_s` (Schwung-Höhepunkt).
- **Clip-Datensatz:** `dataset/_dataset01_01020304/labels.csv` mit 1,2-s-Clips. Anker 700 ms ist der Schwung-Höhepunkt.
- **`release_ms`:** Den echten Abwurf-Zeitpunkt hat `tools/mark_release.py` per Ball-Tracker für 33 der 42 Würfe aus 02/03 gesetzt (610–870 ms). Tim hat das Format abgesegnet.
- **Folds** (nach Aufnahme-Session gruppiert):
  - A = 200828 + 201703
  - B = 200001 + 195808 + 201452
  - C = 170441 + 201622 + d01
  - D = 215305 (nur Test, Ball unsichtbar)

  Die busy-idle-Aufnahme ist in zusammenhängende Zeitblöcke pro Fold geschnitten.
- **Bekannte Schwächen:**
  - Alle Ruhe-Clips im neuen Datensatz stammen aus einer anderen, unruhigen Szene. Das lädt zur Abkürzung „volles Bild = Ruhe“ ein.
  - Insgesamt gibt es nur etwa 42 Würfe mit sichtbarem Ball.
  - Unsicherheiten sind deshalb groß: Allein der Seed macht ±0,03–0,07 AUC aus.

---

## 4. Ablauf der Arbeit (chronologisch)

### 4.1 Opus-„ultracode“-Session (21.11. abends)

Quelle: `noah_sources/handoff_overnight_2026-11-21.md`.

**Panel mit 5 Blickwinkeln** (Labels, Readout, Arch/FPGA, synthetische Daten, Eval-Skeptiker) und einer Synthese:
- Primäre Kennzahl `prim`: **within-fold AUC Wurf vs. Täuschung auf 02/03**. Die gepoolte AUC wird zu 87 % von Ruhe-Clips dominiert.
- Entscheidungsregel: Eine Änderung gilt erst ab Δ ≥ max(0,03; Seed-Rauschen d0) als echt, und mindestens 2 von 3 Folds müssen besser werden.
- Pflicht-Prüfung bei jedem Lauf: Ball entfernen (`cf_erase`) und Ball in eine Täuschung einsetzen (`cf_paste`).
- Experimentliste E1–E10.
- Volltext: `noah_sources/panel_0*.md`.

**Web-Recherche** (`noah_sources/research_0*.md`):
- Alle Systeme, die tatsächlich Bällen ausweichen oder sie fangen, erkennen den **Ball** direkt und prüfen die Physik:
  - Falanga 2020 (Drohne, 3,5 ms Wahrnehmung);
  - Forrai 2023 (ANYmal fängt, RANSAC-Parabel);
  - Zhu 2023 (Mini-Cheetah weicht aus);
  - Delbrucks Roboter-Torwart;
  - EV-Catcher (binäre Eventbilder).
- Laut REBot (Go2) braucht das Ausweichen **mehr als 0,5 s Vorwarnung**.
- Vorarbeit zur Erkennung des **Abwurf-Moments** mit Event-Kamera wurde nicht gefunden. Das ist ein Originalitäts-Argument.

**Gefundene Bugs in Tims `robust_cv.py`** (alle in `runs/ball/ball_cv.py` behoben):
1. Beim zeitlichen Verschieben (±8 Bilder) und Strecken der Fenster bleibt das Ziel bei Frame 70. Es liegt dadurch bis zu ±80 ms falsch, bei 02 bis zu 200 ms vor dem Abwurf.
2. `prep_raw.prep_clips` ignoriert `release_ms`.
3. `counts()` zählt ab dem ersten Event statt ab dem Clip-Start. Ruhe-Clips sind dadurch um bis zu 222 ms verschoben.
4. `idle_idx` spielt Ruhe-Fenster im Ping-Pong teils rückwärts ab. Das gibt es nur im Training.
5. `update_bn` läuft auf augmentierten, also ausgedünnten Daten.
6. Die `ts`-Loss mittelt kumulativ ab dem Fensterstart und verwässert so den Ball-Impuls. `--sym` erzwingt Gleichheit genau in den Bildern, in denen der Ball sichtbar ist (61–69).
7. Eine vertikale Verschiebung um 8 Zellen schiebt den Ball aus dem Bild.

**Klassische Baselines** (CPU):
- **Stolperdraht:** aktive Zellen in den oberen 8 Zeilen, summiert über 2 Bilder. Ergebnis 42/42 Würfe bei 0/65 Täuschungen in 02/03, aber in-sample.
- **Kausaler Ball-Tracker** (`tools/causal_ball_tracker.py`, Ablationen in `runs/ball/tracker/eval_final.txt`):
  - 34/42 Würfe, davon 32/33 innerhalb von 100 ms; im Median 40 ms nach dem Abwurf;
  - **0/104 Täuschungen über alle Sätze**;
  - Ruhe: 0,2/min in der unruhigen Szene, 0,8/min in der Wurf-Szene.
  - Wichtigste Gates laut Ablation: Konstant-Geschwindigkeits-Fit, Residuen-Grenze, Isolationskriterium.

### 4.2 Nacht 21./22.11.

- **Laptop wach halten:** gsettings und ein `systemd-inhibit`-Dienst. Nach dem Kernel-Update-Neustart um 9:30 neu gesetzt, das Zuklappen des Deckels wird jetzt ignoriert.
- **14 Experimente auf dem A100** (`runs/ball/pqueue.sh`, 2 parallel), dazu einige lokal auf der RTX 4070. Ergebnisse in Abschnitt 5.
- **Platinen-Test ohne Kamera:**
  - `tools/board_check.py`: deployt Gewichte (mit Backup), spielt Clips als 0/1-Bilder mit 100 Hz ab und vergleicht mit der riverops-Fixpunkt-Simulation auf identischer Sequenz.
  - `tools/ball_live.py` und `tools/stream_eval.sh`: spielen **ganze Aufnahmen** über das Board ab, jeweils mit dem Fold-Modell, das sie nicht gesehen hat. Die Auslöse-Logik wird gegen `labels.csv` gezählt.
- **Exporter `training/export_ball.py`:**
  - Kopf mit Nullen auf 11 Ausgänge aufgefüllt;
  - Regressionstest reproduziert Tims Board-Gewichte bitgenau (md5 9ff58e0e…);
  - Fixpunkt-Check;
  - neu: `--roi-rows`.
- **ROS:** `go2_sim_tools/ball_trigger_udp.py` übersetzt UDP-JSON in `/reflex/arm|dash|disarm` für `two_stage_reflex`. Der Entry-Point steht in `setup.py`. **Nicht in ROS getestet.**

### 4.3 Morgen 22.11.

- **`final_w2_paste`:** auf allen Daten trainiert (A100), exportiert (md5 d4295de6…), aufs Board gebracht.
- **Live-Anzeige:** `tools/snn_live.py` in `record_tool.py` (http://127.0.0.1:8766) zeigt die Evidenz d, die Schwelle θ, Spikes und das „AUSWEICHEN!“-Overlay. Die Ereignisse landen in `dataset/04/samples/snn_events.csv`.
  - Test mit Echtzeit-Replay von 200828 als falscher Kamera: 3/3 Würfe, höchstens 50 ms nach dem Label, 1 Fehlauslöser, 0 Täuschungen.
- **Datenaugmentation** (`ball_cv.py`, nur per Flag, Backup `runs/ball/old/ball_cv_before_trackaug.py`):
  - `--track-aug`: eingefügte Ballspuren mit Tempo ±30 %, Bogen ±25 %, bis zu 6 Zellen höher;
  - `--p-overlay` / `--overlay-max`: stärkere Überlagerung mit der unruhigen Ruhe-Szene.
  - Ergebnisse in Abschnitt 5.

---

## 5. Ergebnisse: Kreuzvalidierung (Clips, Float-Modell, binäre Eingabe)

Quelle: `runs/ball/compare.py --base w1_win`. 582 gemeinsame zurückgehaltene Clips: 49 Würfe, 68 Täuschungen, 465 Ruhe.

**Spalten:**
- `prim`: within-fold AUC Wurf vs. Täuschung auf 02/03.
- `auc_tf`: gepoolte AUC Wurf vs. Täuschung.
- `rec250`: Anteil der Würfe, die innerhalb von 250 ms nach dem Abwurf auslösen, bei einer Schwelle mit 10 % Täuschungs-Fehlauslösern.
- `busy%`: Anteil der Clips aus der unruhigen Ruhe-Szene, die auslösen.
- `cf_paste`: löst aus, wenn ein Ball in eine Täuschung eingesetzt wird (hoch = gut).
- `cf_erase`: löst noch aus, wenn der Ball aus einem Wurf entfernt wird (niedrig = gut).

| Lauf | Änderung | prim | auc_tf | rec250 | busy% | cf_paste | cf_erase |
|---|---|---|---|---|---|---|---|
| b0_tim | Tims Setup (mit Bugs), nur Folds A+B | ~0,56 | 0,59/0,53 | – | – | 0,22–0,30 | 0 |
| w1_win | Bugs behoben, Fenster-Loss | 0,944 | 0,872 | 0,63 | 0,07 | 0,68 | 0,00 |
| w1_win_s1 | anderer Seed | 0,978 | 0,938 | 0,49 | 0,08 | 0,57 | 0,00 |
| **w2_paste** | + Ball in Täuschungen eingefügt | **0,991** | **0,965** | **0,76** | 0,10 | 0,76 | 0,00 |
| f1_ball | + Einfügen + Entfernen + Hilfskopf | 0,960 | 0,874 | 0,61 | 0,06 | 0,77 | 0,00 |
| f1_ball_s1 | anderer Seed | 0,991 | 0,941 | 0,82 | 0,09 | 0,85 | 0,00 |
| a1_aux | Hilfskopf sagt Ball-Zellen voraus | 0,964 | 0,866 | 0,59 | 0,08 | 0,67 | 0,03 |
| w4_erase | Ball aus Würfen entfernt | 0,928 | 0,791 | 0,12 | 0,03 | 0,19 | 0,00 |
| m1_mil | MIL-Loss | 0,939 | 0,824 | 0,69 | 0,02 | 0,75 | 0,06 |
| d1_drop | Würfe aus 01/04 weggelassen | 0,931 | 0,801 | 0,65 | 0,00 | 0,68 | 0,12 |
| r1_roi16 | Netz sieht nur die oberen 16 Zeilen | 0,951 | 0,660 | 0,29 | 0,03 | 0,25 | 0,00 |
| r1_roi16_s1 | anderer Seed | 0,972 | 0,914 | 0,59 | 0,18 | 0,65 | 0,03 |
| r2_roi24 | nur die oberen 24 Zeilen | 0,988 | 0,795 | 0,76 | 0,03 | 0,80 | 0,09 |
| r3_roi16_ball | ROI 16 + Einfügen/Entfernen | 0,933 | 0,827 | 0,39 | 0,01 | 0,32 | 0,00 |
| x1_trackaug | w2_paste + Spur-Variation | – | 0,950 | rec@f10 0,84 | 0,10 | 0,86 | 0,00 |
| x2_trackaug_ov | x1 + stärkere Überlagerung | – | 0,896 | rec@f10 0,71 | **0,014** | 0,81 | 0,00 |

**Lesart:**
- Die Bugfixes allein heben `prim` von etwa 0,56 auf 0,94–0,98. Seed-Rauschen d0 ≈ 0,034.
- `w2_paste` ist +0,047 besser als `w1_win`, Bootstrap-CI [+0,012, +0,094], in allen 3 Folds besser. Das ist die einzige Änderung, die die Entscheidungsregel klar besteht. Die Begründung: Eine Täuschung mit eingesetztem Ball hat die Armbewegung einer Täuschung, aber das Label „Wurf“. Der Ball ist damit der einzige Hinweis, der zu allen Labels passt.
- **ROI-Modelle** (Netz sieht nur den Luftraum) sind nicht besser als das Vollbild. Für die Demo sind sie außerdem „wack“, wie Noah es nannte. Deshalb wurde das Vollbild gewählt.
- **x2** (mehr Überlagerung) senkt die Fehlauslöser in der unruhigen Szene von 10 % auf 1,4 %, verliert aber Trennschärfe bei Täuschungen. Ohne zweiten Seed ist das keine belastbare Aussage. `w2_paste_s1` ist nicht gelaufen: Der Prozess ist beim Session-Wechsel verloren gegangen.
- `w4_erase` (Ball aus Würfen entfernen) schadet. Vermutlich lernt das Netz, dass Würfe ohne Ball „kein Wurf“ sind, und wird dadurch insgesamt zu vorsichtig.

---

## 6. Ergebnisse: echtes FPGA, ganze Aufnahmen

**Werkzeuge:** `tools/stream_eval.sh w2_paste` und `tools/stream_summary.py`. Quelle: `noah_sources/w2_paste_summary.json`.

**Aufbau:**
- Pro Fold wird das Fold-Modell deployt. Dann laufen die ganzen Aufnahmen dieses Folds durch: Laptop-0/1-Bilder → SPLEAT auf dem Kria → `BallTrigger`.
- d = Summe über 10 Bilder von (Wurf- − Kein-Wurf-Spikes). Auslösen bei d ≥ θ, danach 2 s Pause.
- Treffer: Auslösen im Fenster [Label − 0,4 s, Label + 0,8 s].
- 42 Würfe, 65 Täuschungen, 8,8 min Pausen-Zeit.

| Regel | Würfe | Täuschungen lösen aus | Ruhe-Fehlauslöser |
|---|---|---|---|
| SNN θ=3 | 42/42 | 16/65 | 3,95/min |
| SNN θ=4 | 42/42 | 12/65 | 2,15/min |
| **SNN θ=5** (live-Default) | **42/42** | **8/65** | **1,69/min** |
| SNN θ=6 | 38/42 | 6/65 | 1,36/min |
| SNN θ=8 | 34/42 | 3/65 | 0,45/min |
| nur Tracker (Host, klassisch) | 33/42 | 0/65 | 0,56/min |
| **SNN UND Tracker** (15 Bilder), θ=3–5 | 34/42 | **0/65** | **0,11–0,23/min** |
| SNN ODER Tracker, θ=5 | 41/42 | 8/65 | 2,15/min |

**Hardware-Zahlen:**
- **FPGA gegen Fixpunkt-Simulation** (`board_check.py`, b0-Gewichte, 60 Clips, 9000 Bilder): Spikes 58 vs. 61 bzw. 1891 vs. 1905. 2,6 % der Einträge weichen ab. Entscheidungen pro Clip stimmen in 95–100 % überein.
- **Float gegen Fixpunkt** (`final_w2_paste`): Wurf-Spikes 323 → 264, 91 % der Einträge gleich. Schwellen deshalb **auf dem Board** kalibrieren, nicht auf Float.
- **Latenz:**
  - Board, Feld `latency`: Median 0,5–0,6 ms, p95 3,6 ms.
  - Round-Trip Laptop ↔ Board: Median 1,9 ms.
  - Durchsatz: 100 Bilder/s bei 100 Hz Takt, offline bis 480/s.
- **Gesamtlatenz Sensor → Kommando ist noch nicht gemessen.** Vorschlag: 240-fps-Handyvideo oder eine LED im Kamerabild.

**Vorbehalte:**
- Die Schwellen wurden auf denselben zurückgehaltenen Daten gewählt, die Zahlen sind also leicht optimistisch.
- Das Trefferfenster beginnt 0,4 s vor dem Label. Ein Modell, das auf das Ausholen reagiert, würde bei Würfen also auch „treffen“. Die Täuschungs-Zeile fängt das ab.

---

## 7. System-Architektur (Ball-Trigger, Stand heute Morgen)

```
GenX320 ─MIPI→ Kria: despot_debug_camera ─TCP Events→ Laptop
  Laptop: record_tool.py + snn_live.py (oder tools/ball_live.py)
          Events → binäre 2×80×80-Bilder / 10 ms (wie Trainingsdaten)
          ─TCP CBOR→ Kria spleat-server :3333 → RIVER IP (v1, 11 Ausgänge, ch0 Wurf, ch1 kein Wurf)
          ←─ Spikes pro Bild
          BallTrigger (tools/ball_trigger_logic.py, 54 Unit-Tests)
             d = Σ10 (Wurf − kein Wurf) ≥ θ → arm+dash; Refraktärzeit 2 s;
             Watchdog 200 ms ohne Bilder → Reset, nie Ausweichen auf veralteten Daten;
             Bild ungültig (NaN, Wert ≠ 0/1, Spike auf Padding-Kanal = falsches Modell geladen) → Reset
          optional: Tracker (tools/causal_ball_tracker.py) als UND-Bestätigung (--fire and)
          ─UDP JSON→ ROS: ball_trigger_udp → /reflex/arm|dash|disarm → two_stage_reflex → /api/sport/request → Go2
```

**Warum der Laptop die Bilder baut, nicht despot:**
- despot schickt live **Zählwerte**, trainiert wurde auf 0/1. Daran ist Tims altes 5-Klassen-Modell auf dem Board kollabiert (`runs/tim_fpga/ref5_count` vs. `ref5_binary`).
- despot schickte nur etwa 47 statt 100 Ergebnisse/s, und das in Schüben.

**Alternative:** `tools/spleat_binarize_proxy.py`, ein Proxy auf dem Kria, der die Zählwerte in 0/1 umwandelt. Geschrieben, aber nicht getestet.

**`two_stage_reflex`** (aus der Gazebo/MuJoCo-Sim):
- „arm“ → leicht nach vorn treten;
- „dash“ → mit 1 m/s zurückweichen;
- danach zurück zur Startposition.
- Bewegung nur bei CLEAR-Lidar, jede Bewegung hat ein Zeitlimit.

---

## 8. Panel und Recherche: Kernaussagen

Volltexte in `noah_sources/`. Einen Mitschnitt eines Panel-Talks der Veranstaltung habe ich **nicht**. Gemeint sind hier die Panel-Analysen der Agent-Sessions.

**Panel** (`panel_06_synthesize.md`):
- **Kennzahl:** within-fold AUC Wurf vs. Täuschung auf 02/03. Pro Seed berichten, dazu Bootstrap-CI über Clips.
- **„Winner’s curse“:** Das gewählte Modell ist auf denselben Folds optimiert. Die einzige unverzerrte Prüfung ist eine **frische Aufnahme im Demo-Aufbau**: 10 Würfe, 10 Täuschungen, 20–30 min Ruhe.
- **Deploy-Risiken:** Fixpunkt (+18 % Output-Spikes nach PTQ), Zählwerte statt 0/1, Takt. Alle drei sind inzwischen gemessen bzw. umgangen, siehe Abschnitt 6 und 7.
- **Eval-Skeptiker:**
  - Die Kennzahl soll die tatsächliche Host-Regel simulieren.
  - Vorschlag für eine „DemoCost“, mit Gewichten, die vor dem Blick auf Ergebnisse festgelegt werden: 10·Fehlrate + 10·Täuschungs-FPR + 5·Ruhe-FP/min.
  - Das Seed-Rauschen ist etwa so groß wie die erhofften Effekte. Mit 1 Seed pro Arm ist ein Unterschied erst ab etwa 0,10 nachweisbar.

**Recherche** (`research_0*.md`):
- **Ausweichen und Fangen:** Falanga 2020, Forrai 2023, Zhu 2023 und Delbruck erkennen alle das Objekt direkt. Sie nutzen klassische Physik-Gates (Konstant-Geschwindigkeit bzw. Parabel, „Neuheit“ von Pixeln) und haben extrem kurze Latenz.
- **Readout für SNNs:** Fenster- oder EMA-Summe statt kumulativer Mittelung. MIL-Loss nach Streaming-Keyword-Spotting.
- **Augmentation:** Objekt-Einfügen (copy-paste) als stärkster Hebel bei wenigen Positiven. Bestätigt durch `w2_paste`.
- **Biologische und handgemachte Ansätze:** OMS-Retina, STMD-Detektoren für kleine Ziele. Sie unterdrücken große Körperbewegung und betonen kleine, neu auftauchende Objekte. Das ist der Hintergrund für das Isolations- und Neuheits-Gate im Tracker.
- **Vorwarnzeit:** REBot (Go2) braucht mehr als 0,5 s.

**Demo-Ideen aus der Recherche:**
- Zweistufiger Reflex: beim Ausholen bereitmachen, beim Ball ausweichen, bei einer Täuschung sichtbar entspannen.
- Folie mit dem Ball-Test: Ball entfernen → kein Auslösen; Ball einsetzen → Auslösen.

---

## 9. Neue Aufgabe: Früherkennung „Wurfbewegung“ (Wurf = Täuschung = positiv)

### 9.1 Tims Ansatz (`team-1-user@A100:~/early/`, lokale Kopie ohne Banks in `runs/early_tim/`)

- **Ein Ausgangsneuron**, Readout am Host als EMA (α=0,85 pro 10 ms) mit Schwelle θ. Ziel: 1 ab `onset` ms vor dem Anker (300–500 ms) bis 300 ms danach.
- **Daten:**
  - Positive: eigene Würfe und Täuschungen (Anker = Frame 70 im 1,2-s-Clip).
  - Negative: eigene Ruhe, motion_buddy (Durance), DailyDVS-200, DVS Gesture.
  - Zurückgehaltene Positive: DailyDVS-throw/toss als Test mit fremder Kamera.
- **Training auf zusammengeklebten Streams** (300 Bilder) aus 1-s-Clips verschiedener Quellen, mit Maßnahmen gegen Quellen-Abkürzungen:
  - Dichte-Normalisierung;
  - eine schwache „bed“-Schicht fremder Negativer;
  - Polaritätstausch, Spiegeln, Verschieben, Zoom, zeitliches Strecken;
  - Rauschen und Hot Pixels.
- Leak-Zeitkonstanten auf ≤ 8 Bilder gedeckelt. Optional DANN (Adversarial gegen die Quelle).
- **Eval:** 60-s-Negativ-Streams pro Quelle (Fehlalarme/min). Positive werden nach 3 s zurückgehaltener Negativer eingefügt (Erkennungsrate, Latenz).
- **Tims Zahlen** (`runs_ext/summary.json`, 2 Folds):

  | Lauf | θ | Würfe | Täuschungen | Latenz | Fehlalarme |
  |---|---|---|---|---|---|
  | `r2_ext_fa` | 0,8 | 56 % | 57 % | −310 / −410 ms | 1,5/min eigene Szene |
  | `r2_old_fa` | 0,95 | 52 % | 63 % | −345 / −410 ms | 2/min |

- **Finale Modelle:** `runs_ext_final/final_ext_fa`, `final_old_fa` (Stand 09:20).

### 9.2 Mein Test: Tims Fold-Modelle auf ganzen echten Aufnahmen

**Werkzeug:** `tools/early_stream.py`, Float-Modell auf der lokalen GPU, 0/1-Bilder wie live.

**Aufbau:**
- Jede Aufnahme läuft mit dem Fold-Modell, das sie nicht gesehen hat: Fold 0 = 170441, 201703, 215305; Fold 1 = 195808, 200001, 200828, 201452, 201622.
- Treffer: erster Alarm in [Anker − 1,0 s, Anker + 0,3 s].
- Fehlalarm: Alarm außerhalb von ±1,5 s um jedes gelabelte Ereignis. In 02/03 liegen die Ereignisse dicht, es bleibt nur etwa **1 min** Pausen-Zeit. Die Fehlalarm-Raten sind deshalb grob.
- Ergebnisse: `noah_sources/r2_*_0203.json`.

| Modell, θ | Würfe 02/03 | Täuschungen 02/03 | Latenz (Median) | Fehlalarme in Pausen |
|---|---|---|---|---|
| r2_old_fa 0,80 | 25/42 | 43/65 | −260 / −550 ms | 56/min |
| r2_old_fa 0,95 | 11/42 | 17/65 | −10 / −420 ms | 21/min |
| r2_ext_fa 0,80 | 14/42 | 19/65 | −65 / −410 ms | 22/min |
| r2_ext_fa 0,95 | 8/42 | 14/65 | −120 / −25 ms | 4/min |
| r2_ext_fa 0,80 auf 04/215305 | 10/29 | 17/36 | −560 / −170 ms | 10/min |

**Deutung (teils Annahme, mit Begründung):**
1. **Die Eval auf zusammengeklebten Streams ist zu optimistisch.**
   - Der Positiv-Clip beginnt dort 700 ms vor dem Anker mit einem harten Szenenwechsel von fremden Negativen zur eigenen Szene. Ein Modell, das auf den Wechsel reagiert, zählt als „früh“.
   - Tim mildert das ab: Die Hälfte der Kontexte endet mit eigener Ruhe.
   - Die echten Aufnahmen haben keinen Wechsel. Zwischen den Würfen läuft die Person herum, hebt den Ball auf und holt aus. Genau das sind die realistischen harten Negativen, und dort feuert das Modell oft.
2. **Training ohne echte Pausen aus derselben Szene.** Negativ aus der eigenen Szene ist nur die Ruhe-Aufnahme 211821 (andere Szene, unruhig) plus Clips. Die Pausen-Bewegungen zwischen Würfen in 02/03 (Ball holen, zurückgehen) fehlen als Negative.
3. **Dichte-Normalisierung:** Im Training wird jede Quelle auf 80–500 Events pro Bild ausgedünnt (`keep ≤ 1`), live gibt es keine Ausdünnung. Ist die eigene Szene dichter, sieht das Netz live eine Verteilung, die es nie trainiert hat. **Nicht geprüft.**
4. **`update_bn` läuft auf augmentierten Batches**, mit demselben Effekt wie Bug 5 in Abschnitt 4.1.
5. **Der Anker ist der Schwung-Höhepunkt (700 ms), nicht der Abwurf.** Für das neue Label ist das sinnvoll, weil Täuschungen keinen Abwurf haben. Die Latenz-Zahlen beziehen sich deshalb auf den Höhepunkt.

### 9.3 Verbesserungsvorschläge (nach erwartetem Nutzen)

1. **Eval auf ganzen Aufnahmen als Hauptkennzahl.** `tools/early_stream.py` ist fertig: Fold-Modelle auf ihren zurückgehaltenen Aufnahmen, Erkennungsrate, Latenz und Fehlalarme/min in echten Pausen.

   Für belastbare Fehlalarm-Raten braucht es eine **lange Pausen-Aufnahme im Demo-Aufbau**: 5–10 min mit Herumlaufen, Ball aufheben, Winken.
2. **Training auf echten, durchgehenden Fenstern der eigenen Aufnahmen**, zusätzlich zu den zusammengeklebten Streams.
   - Fenster von 3–5 s direkt aus den 02/03/04-Aufnahmen schneiden.
   - Ziel um **jeden** gelabelten Anker im Fenster setzen (Wurf oder Täuschung).
   - Die Zeit dazwischen ist 0, auch das Ball-Aufheben und Zurückgehen.
   - Ein Bereich um Labels mit „invalid“ wird ignoriert.

   Das gibt die realistischen harten Negativen und entfernt die Szenenwechsel. Die Fold-Trennung bleibt pro Aufnahme.
3. **BN-Statistiken auf unaugmentierten Daten auffrischen.** In `ball_cv.py` heißt das `--bn-clean 1`.
4. **Dichte prüfen:** Events pro Bild der eigenen Aufnahmen gegen den Trainingsbereich 80–500 vergleichen. Notfalls `d_hi` anheben oder live genauso ausdünnen.
5. **Zweistufige Entscheidung:** Mit dem neuen Label wird jede Täuschung absichtlich positiv. Für den Ausweich-Demo-Effekt lässt sich das kombinieren:
   - Früherkennung → „arm“ (bereitmachen, nach vorn treten);
   - Ball-Modell `w2_paste` bzw. Tracker → „dash“.

   Genau dafür ist `two_stage_reflex` gebaut. Beide Modelle passen aber **nicht gleichzeitig** auf die Firmware: v1 hat 11 Ausgänge, aber nur ein Netz.

   Ein gemeinsames Netz mit 3 Ausgängen (Ausholen, Wurf, kein Wurf) wäre möglich. `ball_cv.py --swing` hat einen dritten Ausgang „Ausholen“, dessen AUC gegen Ruhe untrainiert schon bei etwa 0,82–0,89 lag (Lauf `t1_swing`, nur 2 Folds fertig).
6. **Kennzahl mit festen Kosten** (Panel-Vorschlag „DemoCost“), damit θ nicht nachträglich auf die Testdaten optimiert wird.

---

## 10. Was ist Neu und was ist Starter (für die Abgabe)

**Neu (Noahs Seite):**
- `ball_cv.py` mit Bugfixes, Fenster-Loss, Ball-Einfügen/-Entfernen, Hilfskopf, ROI, Spur-Augmentation;
- `make_bank.py`, `ball_tracks.py`, `compare.py`;
- `mark_release.py`, der kausale Tracker;
- `export_ball.py` (Padding auf 11, Fixpunkt-Check);
- `board_check.py`, `ball_live.py`, `stream_eval.sh`, `stream_summary.py`, `early_stream.py`;
- `ball_trigger_logic.py` mit Tests;
- `snn_live.py` und die Anbindung in `record_tool.py`;
- `ball_trigger_udp.py`, dazu `two_stage_reflex` und die Sim-Erweiterungen.

**Von Tim:**
- SNN-Architektur `snn_model.py`;
- Pretraining (DVS Gesture) `best.pt`;
- `robust_cv.py` (Basis von `ball_cv.py`);
- `early_cv.py`.

**Starter / Durance:** `gesture_node`, `action_bridge`, `obstacle_detection`, Firmware und despot. **Durance-Material nicht ins öffentliche Repo.**

---

## 11. Offen, Risiken, nächste Schritte

1. **Demo-Entscheidung zum neuen Label:** Nur Früherkennung? Oder zweistufig mit Früherkennung = arm und Ball = dash? Siehe 9.3 Punkt 5.
2. **Frische Kalibrier-Aufnahme im Demo-Aufbau** (Kamera statisch, Ball sichtbar, Aufbau wie 02/03): etwa 15 min mit 20 Würfen, 20 Täuschungen und 5–10 min Pausen-Bewegung. Danach θ festlegen und Zahlen für die Jury erheben.
3. **Latenz Sensor → Kommando messen**, mit LED oder 240-fps-Video.
4. **ROS-Kette mit dem echten Go2 testen:**
   - `ball_trigger_udp` → `two_stage_reflex` ist nicht in ROS gelaufen;
   - auf dem Kria liegt ROS Humble; `two_stage_reflex` ist bisher nur in der Sim-Workspace.
5. **Reproduzierbarkeit:**
   - `runs/` ist git-ignoriert. `ball_cv.py`, `make_bank.py`, `compare.py` und der Tracker sollten nach `training/` bzw. `tools/` übernommen werden.
   - README ergänzen.
   - **Nichts ist committet.**
6. **Unbelegt:** Ein zweiter Seed für `w2_paste` fehlt. Ob x1/x2 echte Verbesserungen sind, ist offen.

---

## 12. Dateien und Befehle

| Was | Wo |
|---|---|
| Training Ball-Modell | `runs/ball/ball_cv.py` (Basis Tims `robust_cv.py`), Bank `runs/ball/bank_mix.pt` (1,1 GB, `make_bank.py`), Spuren `tracks.npz` |
| Vergleich | `cd runs/ball && python compare.py --base w1_win` |
| Läufe / Logs | `runs/ball/runs_ball/<tag>/`, `runs/ball/logs/`, A100: `~/ball_noah/ball/` |
| Export | `CUDA_VISIBLE_DEVICES= python training/export_ball.py --ckpt runs/ball/runs_ball/final_w2_paste/final.pt --out runs/ball/deploy/final_w2_paste --check 60` |
| Deploy | `scp runs/ball/deploy/<x>/weights.txt root@192.168.77.1:/root/weights.txt && ssh root@192.168.77.1 systemctl restart spleat-server` (Backup von Tims Gewichten: `/root/weights_backup_9ff58e0e5671e9af29e954ecda726a70.txt`) |
| Board-Check | `python tools/board_check.py --ckpt <fold.pt> --export <deploydir> --n 60` |
| Stream-Test auf dem Board | `tools/stream_eval.sh w2_paste && python tools/stream_summary.py w2_paste` |
| Live mit Kamera | `python tools/record_tool.py` (Webseite :8766, SNN-Panel) bzw. `python tools/ball_live.py --theta 5 [--fire and] [--udp host:5555]` |
| ROS | `ros2 run go2_sim_tools ball_trigger_udp` → `two_stage_reflex` |
| Früherkennung auf ganzen Aufnahmen | `python tools/early_stream.py --ckpt <model.pt> --rec <bins> --out <dir>`, dann `--score '<dir>/*.npz'` |
| Tims Früherkennung | A100 `~/early/` (lokal ohne Banks: `runs/early_tim/`) |
| Python-Umgebung | `~/miniconda3/envs/river` (riverops, river_tools; scipy nachinstalliert) |
