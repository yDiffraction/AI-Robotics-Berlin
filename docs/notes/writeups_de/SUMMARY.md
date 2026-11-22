# Gesamt-Zusammenfassung: Durance Vision Lab @ Berlin HackLab Robotics

Event-Kamera (GenX320) → SNN auf dem Kria-FPGA → Host-Readout → ROS 2 / Sport-API → Unitree Go2.

Stand: 22.11.2026, ca. 10:35. Erstellt von der Zusammenfassungs-Session aus allen auffindbaren Write-ups und Notizen (Quellen in Abschnitt 8).

**Lesehilfe:**
- **[gemessen]** heißt: Zahl aus einem Lauf, mit Quelle.
- **[Annahme]** heißt: Schätzung, Vermutung oder Rechnung ohne eigene Messung.
- „OOF“ heißt out-of-fold: Das Modell hat die Testdaten nicht gesehen.

> **Wichtig vorab:** Als eigenes Write-up in `docs/writeups/` liegt bisher nur Noahs [`noah_ball_trigger.md`](noah_ball_trigger.md). Für **Laurenz** und **Tim** habe ich ihre READMEs auf den Git-Branches und ihre Dateien auf dem A100 ausgewertet. Für Tims Früherkennung (`~/early`) gibt es **keine** schriftliche Beschreibung, nur Code, `summary.json` und Noahs Auswertung. Wenn eigene Write-ups von Tim oder Laurenz kommen, muss diese Datei nachgezogen werden.

---

## 1. Kurzfassung

### 1.1 Die Arbeitsstränge

| Strang | Wer | Frage, die das Netz beantwortet | Stand |
|---|---|---|---|
| **A: Ball-Trigger** | Noah (Agent-Sessions) | Ist gerade ein **Ball** geflogen? Täuschung = nein. | Modell `final_w2_paste` lief am Morgen live auf dem FPGA. Ausführlich getestet. |
| **B: Wurfbewegungs-Detektor `throwdet`** | Laurenz (Branch `throw-detector`) | Findet **jetzt** eine Wurfbewegung statt? Wurf und Täuschung = ja. | 6 Kandidaten-Modelle, eigene 2-Klassen-Firmware, Live-Ansicht auf dem Kria. Seit 09:23 committet. |
| **C: Früherkennung `early_cv`** | Tim (A100 `~/early`) | Wie B, aber möglichst früh (Ziel ab 300–500 ms vor dem Höhepunkt). | Finale Modelle `final_ext_fa` und `final_old_fa` (Stand 09:20). Nicht exportiert, nicht deployt. |
| **D: Wurf gegen Täuschung (robust)** | Tim (Branch `Tim`, `experiments/robust_release/`) | Wurf oder Täuschung? | Über Sessions hinweg nahe Zufall. Abgelöst durch A. |
| **E: Pretraining** | Tim (`training/pretrain/`) | DVS Gesture, 11 Klassen | Backbone für A–D. |
| **F: Ausweichen / Robotersteuerung** | Tim (`sim/`, `sim/deploy/`, `sim/avoid/`); Laurenz und Noah (Gazebo/MuJoCo, `two_stage_reflex`, `tools/go2_reflex.py`) | – | Sehr gründlich in der Simulation getestet. **Kein dokumentierter Test am echten Go2.** |
| **G: Daten** | Noah (Aufnahmen, Labels, `_dataset01_*`); Laurenz (Augmentierung, Synth, Video, DailyDVS-Import) | – | `_dataset01_01020304`: 77 Würfe, 104 Täuschungen, 288 Ruhe-Clips. |

### 1.2 Was funktioniert, was ist deployt

1. **Die FPGA-Kette funktioniert in zwei unabhängigen Varianten.** In beiden baut der Laptop 0/1-Bilder aus den Kamera-Events und schickt sie an den SPLEAT-Server auf dem Kria.
   - **Variante A (Noah):** 11-Ausgangs-Firmware, Q2.6-Gewichte. Ein neues Modell wird nur durch Tausch der `weights.txt` deployt. Durchsatz 100 Bilder/s, Modell-Latenz 0,5–0,6 ms, Round-Trip 1,9 ms **[gemessen]**.
   - **Variante B (Laurenz):** eigene 2-Klassen-Firmware (riversdk-Job #43), Q3.5-Gewichte. Durchsatz 100 Bilder/s, Round-Trip 2,3–2,6 ms. FPGA und Float-Modell liefern auf dem 01-Stream 27 gegen 27 Alarme, 26 davon innerhalb von 0,3 s **[gemessen]**.
2. **Deployt war:**
   - Noah: `final_w2_paste` mit Live-Anzeige in `record_tool.py`.
   - Laurenz: um ca. 09:21 seine 2-Klassen-Firmware, laut seinem TL;DR mit `D_e10_fold3`, plus `live_view.py`.
   - **Was jetzt gerade auf dem Board läuft, ist unklar** (siehe Abschnitt 4, Punkt W1). Der Kria war bei meinem Check um 10:30 nicht erreichbar.
3. **Am Go2 ist nichts belegt.** Kein Write-up dokumentiert einen Ausweich-Test am echten Hund, weder über die Sport-API (`tools/go2_reflex.py`) noch über den Low-Level-Controller (`sim/deploy/go2_modes.py`).
4. **Die Latenz Sensor → Kommando hat noch niemand gemessen.** Die Jury verlangt sie ausdrücklich.

### 1.3 Beste Zahlen je Aufgabe

| Aufgabe | Bestes Ergebnis | Art der Messung | Quelle |
|---|---|---|---|
| Ball-Wurf erkennen, Täuschung ignorieren (alte Aufgabe) | SNN θ=5: 42/42 Würfe, 8/65 Täuschungen, 1,69 Fehlauslöser/min. **SNN UND Tracker: 34/42, 0/65, 0,11–0,23/min** | echtes FPGA, ganze zurückgehaltene Aufnahmen. θ auf denselben Daten gewählt. | `noah_ball_trigger.md` Abschn. 6, `noah_sources/w2_paste_summary.json` |
| Wurfbewegung erkennen (neue Aufgabe) | e10: AUC 0,865–0,924, Recall bei 2 Fehlalarmen/min 0,54–0,79 (4 Seeds). e26: AUC 0,881–0,919, Recall 0,58–0,82. Alarm ca. 150–180 ms **vor** dem Schwung-Höhepunkt. | Float, OOF auf Clips plus 5,8-min-Ruhe-Stream, 4 Folds | `throwdet/README.md`, `throwdet/results_summary.md` (Branch `throw-detector`) |
| Wurfbewegung, ungesehene Aufnahmen 01 | 23/25 Schwünge (e10-CV-Modell), dazu 3 weitere Alarme in 55 s | Float, zusammengeklebter Stream aus 25 Kurz-Aufnahmen | ebd. |
| Früherkennung (Tim) | 52–56 % der Würfe, 57–63 % der Täuschungen bei 1,5–2 Fehlalarmen/min, Median −310 bis −410 ms | Float, zusammengeklebte Streams, 2 Folds | A100 `~/early/runs_ext/summary.json` |
| Früherkennung (Tim) auf ganzen Aufnahmen | 8–25 von 42 Würfen, 4–56 Fehlalarme/min | Float, ganze Aufnahmen 02/03, OOF. Fehlalarm-Basis nur ca. 1 min Pause. | `noah_ball_trigger.md` Abschn. 9.2, `noah_sources/r2_*_0203.json` |
| Ausweichen (Sim) | SAFE-Dodge: 0/500 Fehlschläge unter harten Bedingungen, 30 cm nach 0,58 s (Median) | MuJoCo, Low-Level-Policy | Branch `Tim`, `sim/README.md`, `sim/deploy/README.md` |
| Sicherheit (Sim/Fake-Robot) | Supervisor 54/54 Szenarien, Moduswechsel 12/12, Hindernis-Suite 290 Läufe ohne Sturz | Fake-Go2 über echtes DDS bzw. MuJoCo | ebd., `sim/avoid/README.md` |

---

## 2. Die Stränge im Einzelnen

### 2.1 Strang A: Ball-Trigger (Noah)

Details stehen in [`noah_ball_trigger.md`](noah_ball_trigger.md).

**Kernaussage:** Wurf und Täuschung unterscheiden sich nur durch den **Ball im Luftraum** über dem Werfer, nicht durch die Armbewegung. Ein Großteil von Tims schwachen Ergebnissen kam von 7 Bugs im Trainingscode, vor allem vom zeitlich falsch liegenden Ziel.

**Bestes Modell `w2_paste`:** Echte Ballspuren werden im Training in Täuschungen eingefügt und als „Wurf“ gelabelt.
- Within-fold-AUC Wurf gegen Täuschung 0,991, gepoolt 0,965 **[gemessen, Float, OOF-Clips]**.
- Einziges Experiment, das die vorab festgelegte Entscheidungsregel klar besteht: +0,047 gegenüber `w1_win`, CI [+0,012; +0,094].
- Ball-Test **[gemessen]**: Ohne Ball löst es nie aus (0 %), mit eingesetztem Ball in 76 % der Fälle.

**Klassische Vergleiche (Host-CPU):**
- Kausaler Tracker: 34/42 Würfe, 0/104 Täuschungen, Median 40 ms nach dem Abwurf **[gemessen, auf denselben Daten getuned]**.
- Stolperdraht: 42/42 Würfe, 0/65 Täuschungen **[in-sample]**. In der unruhigen Szene aber 23 Fehlauslöser/min.

**Grenze:** Funktioniert nur, wenn der Ball nach dem Abwurf im Bild ist. In 04/215305 ist er das nicht. Gebraucht wird ein Aufbau wie 02/03: Kamera statisch, Kopf des Werfers nicht am oberen Bildrand.

### 2.2 Strang B: Wurfbewegungs-Detektor `throwdet` (Laurenz)

Quelle: Branch `throw-detector`, `throwdet/README.md` und `throwdet/results_summary.md`; A100 `~/throwdet/`.

**Aufbau:**
- v1-Backbone mit 2 Ausgängen, Readout am Host als EMA mit Haltezeit und 1 s Refraktärzeit.
- Ziel: 1 von 250 ms vor dem Schwung-Höhepunkt bis 300 ms danach.

**Daten:**
- Eigene Clips: 77 Würfe, 104 Täuschungen, 288 Ruhe-Clips aus 211821, dazu **255 Pausen-Clips zwischen den Würfen** aus `_dataset01_010203`.
- Eigene Augmentierung und Splices.
- Synth v2 und Handy-Videos.
- DailyDVS-200 mit 47 Personen.
- Gegen Abkürzungen über die Datenquelle: Batches nach (Quelle, Ziel) ausbalanciert, Dichte angeglichen, Hintergrund eingeblendet.

**Ergebnisse** (4-fach gruppierte CV nach Aufnahme, 37 Läufe):
- Bestes Rezept e10 bzw. e26, siehe Tabelle 1.3 **[gemessen, Float, OOF]**.
- Seed-Rauschen: bis 0,06 AUC und bis 0,25 Recall.
- Externe Daten helfen bei neuen Aufnahmen. Ohne sie sinkt der Fold mit 04 auf AUC 0,75. Ohne Synth/Video werden beide Seeds schlechter als jeder e10-Seed.
- Pro Aufnahme beim Fold-3-Modell (`D_e10_fold3/cv_readout.txt`, Arbeitspunkt 2/min):
  - 03: 100 % der Schwünge erkannt;
  - 02/170441: nur 20 %;
  - 04/215305: 56–64 %;
  - Pausen-Clips in 02/03 lösen in 0–25 % der 1,2-s-Clips aus (je nach Aufnahme).
- Fixpunkt: PTQ behält 94–98 % der Alarm-Entscheidungen bei.

**Werkzeuge:**
- `deploy_board.sh` mit Backup und Restore;
- `live_view.py` mit Backend Board oder Laptop;
- `replay.py`;
- `cut_recording.py` und `adapt.sh` für die Anpassung an den Ort (5–10 min auf dem A100).

**Empfohlene Test-Reihenfolge laut Laurenz:** D, E, A, B, C. Auf dem Board `E_moredd_q35` statt `E_moredd`.

### 2.3 Strang C: Früherkennung `early_cv` (Tim)

Quelle: A100 `~/early/` (lokale Kopie `runs/early_tim/`), Auswertung in `noah_ball_trigger.md` Abschn. 9.

**Aufbau:**
- 1 Ausgang, EMA-Readout, Ziel ab `onset` (300–500 ms) vor dem Anker.
- Training auf zusammengeklebten 3-s-Streams aus eigenen Clips, motion_buddy (Durance), DailyDVS und DVS Gesture.

**Ergebnisse, Iteration 1** (4 Folds, `runs_early/summary.json`) **[gemessen, Float, Stream-Eval]**:
- `e1_onset500`: 69 % der Würfe, 71 % der Täuschungen, Median −470 ms.
- Dafür aber 14 Fehlalarme/min auf der eigenen Ruhe-Szene.

**Ergebnisse, Iteration 2** (2 Folds, `runs_ext/summary.json`):
- `r2_ext_fa`: θ=0,8, 56 %/57 %, 1,5/min eigene Szene, aber 18/min auf DailyDVS-Negativen.
- `r2_old_fa`: θ=0,95, 52 %/63 %, 2/min.
- `x_*` und `r3_*` liegen darunter oder gleichauf.

**Noahs Test auf ganzen Aufnahmen** (gleiche Fold-Modelle):
- Bei θ=0,95: 8–11 von 42 Würfen, 4–21 Fehlalarme/min.
- Vermutete Ursachen **[Annahme mit Begründung]**:
  - harter Szenenwechsel im zusammengeklebten Eval;
  - keine echten Pausen-Bewegungen als Negative;
  - BN-Statistiken auf augmentierten Daten;
  - Ausdünnung nur im Training.

### 2.4 Strang D: Wurf gegen Täuschung, robust (Tim, Branch `Tim`)

Quelle: `experiments/robust_release/README.md`.

- Mit dem alten Datensatz (Würfe und Täuschungen in Blöcken aufgenommen) hat das Netz eine Abkürzung über die Aktivität vor dem Abwurf gelernt. Prefix-Symmetrie und Erzeugung aus echten Aufnahmen beheben das: 0,85 bei n=24.
- Neuer Datensatz, Folds nach Session: 0,50–0,56 balanced accuracy. Mit horizontaler Verschiebung 0,62–0,69 (2 Seeds).
- Folds auf Clip-Ebene: 0,89. Das ist optimistisch, weil dieselben Sessions in Training und Test vorkommen.
- Ein einziges handgemachtes Ball-Merkmal (obere 8 Zeilen) schafft 0,91.
- Tims eigener Schluss ist derselbe wie in Strang A: Das Signal ist der Ball.

### 2.5 Strang E: Pretraining (Tim)

- DVS Gesture, v1-Backbone im Board-Format.
- Zurückgehaltene Nutzer: 87,9 % pro Fenster, 91,3 % pro Aufnahme **[gemessen, Float]**.
- Wird von A, B und C als Start genutzt.
- Laut Laurenz hilft es bei `throwdet` nicht messbar über das Seed-Rauschen hinaus (e03 gegen e01).

### 2.6 Strang F: Ausweichen und Sicherheit

**Tim (Branch `Tim`, alles Simulation bzw. Fake-Robot):**
- `sim/README.md`, Walk-these-ways-Policy in MuJoCo:
  - Der SAFE-Dodge ist in 500 harten Läufen ohne Fehlschlag. 30 cm nach 0,58 s (Median), p95 0,64 s.
  - Die Policy schafft höchstens etwa 0,42–0,44 s bis 30 cm.
- `sim/deploy/README.md`:
  - Supervisor mit E-Stop (L2+B), Sturz- und Watchdog-Erkennung, Damping.
  - 54/54 Szenarien gegen Fake-Go2 über echtes DDS; Stopp-Simulation 0/100 umgekippt mit kd=5; Moduswechsel 12/12.
  - **Bei der Auslösung über `go2_modes.py`:** zuerst 0,5 s Stehen, dann etwa 2 s Traben auf der Stelle, dann seitlich. Außerdem wird das Ausweichen verweigert, wenn der Hund nicht mindestens 1 s still stand.
- `sim/avoid/README.md`: Lidar-Wächter mit Nav2 Collision Monitor, 290 Läufe ohne Sturz. Bekannter blinder Fleck neben den Hinterbeinen.

**Noah / Laurenz:**
- `two_stage_reflex` in der Gazebo-Demo über die Sport-API (`runs/two_stage/all_runs.log`, 11 Versuche):
  - von der Entscheidung bis zur Rückwärtsbewegung im Median 565 ms;
  - bis −10 cm 730 ms;
  - bis −25 cm 952 ms;
  - 10 von 11 virtuellen Bällen ausgewichen **[gemessen, Sim-Zeit, Echtzeitfaktor 1,0]**.
- `tools/go2_reflex.py`: derselbe zweistufige Reflex direkt über die Sport-API vom Laptop aus. Ein Ergebnis am echten Hund ist nirgends dokumentiert.

---

## 3. Vergleich der Ansätze

**Achtung, die Zahlen sind nur eingeschränkt vergleichbar.** Die Spalten „Aufgabe“ und „Messart“ sagen, was gemessen wurde.

| Ansatz | Aufgabe | Messart | Würfe erkannt | Täuschungen lösen aus | Fehlauslöser Ruhe | Zeitpunkt | Quelle |
|---|---|---|---|---|---|---|---|
| Ball-SNN `w2_paste` θ=5 | Ball | **FPGA**, ganze Aufnahmen 02/03, OOF-Folds, θ in-sample | 42/42 | 8/65 (soll niedrig sein) | 1,69/min (8,8 min Pausen) | nach dem Abwurf (Trefferfenster −0,4…+0,8 s um das Label) | N Abschn. 6 |
| Ball-SNN UND Tracker | Ball | wie oben | 34/42 | 0/65 | 0,11–0,23/min | Tracker im Median +40 ms nach dem Abwurf | N Abschn. 6 |
| Kausaler Tracker allein (klassisch, Host) | Ball | Host, ganze Aufnahmen, auf denselben Daten getuned | 33–34/42 | 0/65 (0/104 mit 04) | 0,56/min (Board-Lauf); 0,2–0,8/min (CPU-Eval) | +40 ms (Median) | N Abschn. 4.1, 6 |
| Stolperdraht (8 Zeilen) | Ball | in-sample | 42/42 | 0/65 | 0,6/min Wurf-Szene, **23/min** unruhige Szene | ca. +90 ms | Handoff 1.1 |
| `throwdet` e10 / e26 (Laurenz) | Bewegung | Float, OOF-Clips plus 5,8-min-Ruhe-Stream | Recall 0,54–0,79 / 0,58–0,82 (Würfe und Täuschungen zusammen) | **positiv gewollt** | 2/min (Arbeitspunkt, Ruhe-Aufnahme 211821) | ca. −150 bis −180 ms vor dem Höhepunkt | L README |
| `throwdet` D, 01-Stream | Bewegung | Float, 55 s, 01 ungesehen, zusammengeklebt | 24/25 (CV-Modell e10: 23/25) | positiv gewollt | 3 weitere Alarme in 55 s | −0,6…+0,3 s Fenster | L README |
| `throwdet` auf dem FPGA (Modell im README nicht genannt, laut TL;DR vermutlich D) | Bewegung | **FPGA**, 01-Stream | 27 Alarme (Float auch 27) | – | – | – | L README „Checks“ |
| Tim `r2_ext_fa` θ=0,8 | Bewegung | Float, zusammengeklebte Streams, 2 Folds | 56 % (Täuschungen 57 %) | positiv gewollt | 1,5/min eigene Szene, 18/min DailyDVS | −310 / −410 ms | T `runs_ext/summary.json` |
| Tim `r2_ext_fa` θ=0,8 | Bewegung | Float, **ganze Aufnahmen** 02/03, OOF | 14/42 (Täuschungen 19/65) | positiv gewollt | 22/min (nur ca. 1 min Pause) | −65 / −410 ms | N Abschn. 9.2 |
| Tim `r2_old_fa` θ=0,95 | Bewegung | Float, ganze Aufnahmen 02/03 | 11/42 (17/65) | positiv gewollt | 21/min | −10 / −420 ms | N Abschn. 9.2 |
| Tim `robust` (Session-CV) | Wurf gegen Täuschung | Float, Clips | balanced acc 0,62–0,69 | – | – | Fensterende | T `experiments/robust_release/README.md` |

**Warum die Zeilen nicht direkt vergleichbar sind:**
1. **Aufgabe:** Bei „Ball“ ist eine auslösende Täuschung ein Fehler, bei „Bewegung“ ist sie gewollt. Erkennungsraten der beiden Gruppen lassen sich nicht gegeneinander stellen.
2. **Clips gegen ganze Aufnahmen:**
   - Laurenz und Tim messen auf Clips bzw. zusammengeklebten Streams.
   - Noah misst auf durchgehenden Aufnahmen mit echten Pausen (Ball aufheben, zurückgehen).
   - Tims Zahlen fallen beim Wechsel auf ganze Aufnahmen stark ab. Für `throwdet` gibt es diesen Test **noch nicht**.
3. **Float gegen FPGA:** Nur die Zeilen zum Ball-SNN und `throwdet`-auf-FPGA sind auf echter Hardware gemessen. Float und Fixpunkt stimmen in beiden Strängen in 91–98 % der Entscheidungen bzw. Einträge überein. Schwellen sollten deshalb auf dem Board kalibriert werden.
4. **In-sample gegen held-out:**
   - Stolperdraht und Tracker-Parameter sind in-sample.
   - Beim Ball-SNN sind die Modelle OOF, aber θ wurde auf denselben Aufnahmen gewählt.
   - Bei `throwdet` wurde der Readout OOF gewählt. Die finalen Modelle A/B/C/E sind aber auf allen Daten trainiert, ihre Schwelle stammt aus den CV-Modellen. Laurenz sagt selbst, dass sie nicht exakt übertragbar ist.
5. **Fehlalarm-Basis:**
   - Laurenz: 5,8 min Ruhe-Aufnahme 211821, andere Szene ohne Werfer.
   - Noah (Ball): 8,8 min Pausen in den Wurf-Aufnahmen.
   - Noah (Tim-Test): ca. 1 min Pausen.
   - Tim: zusammengeklebte 60-s-Streams pro Quelle.
6. **Zeitbezug:** Noahs Ball-Zahlen beziehen sich auf das Label bzw. den Abwurf. `throwdet` und `early` beziehen sich auf den Schwung-Höhepunkt (Anker bei 700 ms). Dieser liegt laut `mark_release.py` grob dort, wo auch der Abwurf liegt (610–870 ms im Clip).

---

## 4. Widersprüche und welche Zahl belastbarer ist

**W1: Was läuft auf dem Board?**
- Noahs Write-up nennt `final_w2_paste` als deployt, auf der 11-Ausgangs-Firmware.
- Laurenz hat um ca. 09:21 mit `deploy_board.sh` die **2-Klassen-Firmware** installiert. Sein Backup `20261122_092155` nennt er „gesture demo“.
- `w2_paste` braucht die 11-Ausgangs-Firmware. Laurenz' Modelle brauchen Job #43. Beide Modelle passen **nicht** gleichzeitig aufs Board.
- **Klären, bevor jemand live testet.** Rückweg zu A: `deploy_board.sh --restore 20261122_092155`, danach md5 der `weights.txt` gegen `d4295de6…` prüfen.

**W2: Tims Früherkennung, Clips gegen ganze Aufnahmen.**
- Clips: 52–56 % bei 1,5–2/min. Ganze Aufnahmen: 8–25 von 42 bei 4–56/min.
- **Belastbarer ist der Test auf ganzen Aufnahmen**, weil er dem Live-Betrieb entspricht. Die Fehlalarm-Raten dort sind aber grob, weil nur ca. 1 min Pause zur Verfügung stand.

**W3: `throwdet` „ca. 70 % bei 2/min“ gegen den realen Betrieb.**
- Die Fehlalarm-Rate kommt aus einer Ruhe-Aufnahme ohne Werfer.
- Laurenz' eigene Zahlen pro Aufnahme zeigen: Pausen-Clips aus 02/03 lösen in 0–25 % der 1,2-s-Clips aus.
- **[Annahme]** In echten Wurf-Pausen dürfte die Fehlalarm-Rate deshalb über 2/min liegen. Gemessen ist das nicht.
- Laurenz nennt als Hauptquelle von Fehlalarmen selbst „Personen nah an der Kamera“.
- **Belastbare Zahl fehlt.** Ein Lauf von `tools/early_stream.py` (oder `throwdet/replay.py`) mit den OOF-Fold-Modellen auf den ganzen Aufnahmen 02/03/04 würde sie liefern.

**W4: Ruhe-Fehlalarme des Trackers: 0,2–0,8/min oder 0,56/min.**
- Verschiedene Läufe: CPU-Eval pro Szene gegen Board-Stream-Eval. Kein echter Widerspruch.
- Beide sind auf den Tuning-Daten gemessen.

**W5: Label-Historie.** Je nach Zeitpunkt bedeutet „positiv“ etwas anderes.
- 21.11. 22:30 (Tim, `robust`): Wurf gegen (Täuschung + Ruhe).
- Noahs Ball-Trigger: dieselbe Definition.
- `throwdet` (Laurenz, über Nacht) und seit 22.11. ca. 10:20 die Teamvorgabe: (Wurf + Täuschung) gegen Ruhe.
- Ältere Zahlen sind daher nicht auf die neue Aufgabe übertragbar.

**W6: Zeitbudget für das Ausweichen.**
- Laurenz rechnet: Alarm ca. 150 ms vor dem Höhepunkt plus ca. 0,3 s Flugzeit, also beginnt das Ausweichen ca. 0,45 s vor dem Aufprall. Die Flugzeit ist **[Annahme]**: 2–3 m bei 8 m/s.
- Gemessen sind bisher nur Simulationszeiten:
  - Sport-API (Gazebo): 565 ms bis zur Rückwärtsbewegung, 952 ms bis −25 cm.
  - Low-Level-SAFE-Dodge (MuJoCo): 0,58 s bis 30 cm. In `go2_modes.py` gehen dem aber 0,5 s Stehen und ca. 2 s Traben voraus.
- **Folgerung [Annahme]:**
  - Ein Ausweichen, das erst **nach** dem Abwurf ausgelöst wird (Strang A), kommt bei einem direkten Wurf aus 2–3 m zu spät.
  - Rechtzeitig ist nur eine Auslösung beim Ausholen (B/C) mit bereits vorbereitetem Hund (armed, trabend) oder ein hoher, langsamer Bogenwurf.
  - Das spricht klar für den zweistufigen Reflex.

**W7: Recherche-Aussage „> 0,5 s Vorwarnung“ (REBot).** Passt zu W6. Sie ist eine Literatur-Angabe, für unseren Hund nicht gemessen.

---

## 5. Empfehlung für Demo und Jury-Story

Dies sind Vorschläge der Zusammenfassungs-Session. Die Entscheidung liegt beim Team.

### 5.1 Demo-Aufbau (Working System, 35 Punkte)

1. **Ein Netz auf dem FPGA: `throwdet`** (Laurenz, z. B. `D_e10_fold3`) mit der 2-Klassen-Firmware.
   - Es passt zur neuen Aufgabe und ist auf dem Board verifiziert (27/27 Alarme wie Float).
   - Die Live-Ansicht existiert, ebenso Werkzeuge zum Anpassen an den Ort.
2. **Zweistufiger Reflex:**
   - Stufe 1: `throwdet`-Alarm → **arm** (Hund lehnt sich vor oder trabt an).
   - Stufe 2, optional, nur wenn die Story „Täuschung erkennen“ bleiben soll: kausaler Ball-Tracker auf dem Host → **dash**, sonst nach Timeout **disarm**.
   - Der Tracker ist klassisch und muss als „Bestätigung auf dem Host, kein SNN“ deklariert werden.
   - Das Ball-SNN `w2_paste` passt nicht gleichzeitig aufs Board (W1). Ein gemeinsames 3-Ausgangs-Netz (Ausholen / Wurf / kein Wurf) ist nicht trainiert.
3. **Fallback, wenn die Zeit fehlt:** nur Stufe 1. Jeder Alarm löst direkt das Ausweichen aus, Täuschungen eingeschlossen. Das entspricht genau der neuen Teamvorgabe.
4. **Aufbau:** Kamera statisch auf Stativ, Werfer ca. 3 m entfernt, über dem Kopf freie Fläche (wie 02/03, **nicht** wie 04). Keine Zuschauer nah vor der Kamera, das ist laut Laurenz die Hauptquelle von Fehlalarmen.
5. **Vorher am Echt-Hund testen:**
   - die Kette Laptop → Sport-API (`tools/go2_reflex.py`) bzw. `ball_trigger_udp` → `two_stage_reflex`;
   - die Zeit von der Entscheidung bis zur Bewegung.

### 5.2 Robustness (25 Punkte): Zahlen, die wir zeigen können bzw. noch erheben müssen

- **Schon da:**
  - Ball-Strang auf dem FPGA mit ganzen zurückgehaltenen Aufnahmen: SNN allein 8/65 Täuschungen, SNN UND Tracker 0/65 bei 34/42 Würfen.
  - `throwdet`: Seed-Streuung über 4 Seeds offen berichtet; Ablationen zum Wert externer Daten; DailyDVS-Personen als Holdout.
- **Messbare Verbesserung (Level 3):**
  - Tims Ausgangs-Setup `b0_tim` hatte AUC 0,53–0,59, nach Bugfixes und Ball-Einfügen 0,965 (Ball-Aufgabe).
  - `throwdet` gegen die Aktivitäts-Baseline: balanced accuracy 0,77 gegen AUC um 0,9. Die Maße sind verschieden, das ist ehrlich als „verschiedenes Maß“ zu kennzeichnen.
- **Fehlt noch:**
  - eine **frische Aufnahme im Demo-Aufbau** (ca. 15 min: 20 Würfe, 20 Täuschungen, 5–10 min Pausen-Bewegung), auf der θ **vor** dem Test festgelegt wird;
  - darauf Erfolge pro Versuch und Fehlalarme/min ohne erneutes Nachjustieren.
  - Das ist der einzige unverzerrte Test (Panel, „Winner's curse“).

### 5.3 Latenz (von der Jury ausdrücklich verlangt)

- Getrennt berichten:
  - Inferenz auf dem FPGA: 0,5–0,6 ms (Noah) bzw. Round-Trip 1,9–2,6 ms;
  - Entscheidungszeitpunkt relativ zum Höhepunkt (`throwdet` ca. −150 ms);
  - **Sensor → Kommando**, noch nicht gemessen. Vorschlag: LED im Kamerabild oder 240-fps-Handyvideo.

### 5.4 Originality (20 Punkte)

- Recherche (`noah_sources/research_*`): Zur **Erkennung des Wurf- bzw. Abwurf-Moments mit Event-Kamera** wurde keine Vorarbeit gefunden. Bestehende Ausweich-Systeme (Falanga, Forrai, Zhu, Delbruck) erkennen alle das fliegende Objekt.
- „Bereitmachen beim Ausholen, Ausweichen beim Ball, sichtbar entspannen bei einer Täuschung“ ist als Demo-Moment stark.
- Folie zum Ball-Test aus Strang A: Ball entfernen → nie Auslösen; Ball in Täuschung einsetzen → 76 % Auslösen. Das zeigt, *worauf* das Netz schaut.

### 5.5 Engineering und Reproducibility (je 10 Punkte)

- Sicherheits-Engineering aus Strang F: Supervisor, E-Stop, Watchdogs, 54/54 Tests. Lohnt eine Folie, auch wenn es nur in der Simulation getestet ist.
- **Repo-Zustand:**
  - `main` steht noch auf dem Stand vom 21.11. 13:04.
  - Laurenz' Arbeit ist auf `throw-detector` committet.
  - Tims `experiments/robust_release`, `pretrain`, `sim` sind auf `Tim`.
  - **Noahs gesamte Arbeit seit gestern Abend ist nicht committet** (`runs/` ist git-ignoriert).
  - **Tims `~/early` ist nur auf dem A100.**
  - Vor der Abgabe alles auf `main` zusammenführen, README mit einem Befehl pro Schritt. Durance-Material (Firmware-Tarballs, motion_buddy, Starter-Code) draußen lassen.

---

## 6. Offene Punkte

Verantwortliche nur, wo aus den Quellen erkennbar. Sonst steht „offen“.

| # | Punkt | Verantwortlich | Quelle |
|---|---|---|---|
| 1 | Board-Zustand klären (W1) und festlegen, welches Netz in die Demo kommt | Noah + Laurenz | N Abschn. 4.3, L README „Deploy“ |
| 2 | Entscheidung: nur Stufe 1 (Bewegung → ausweichen) oder zweistufig mit Ball-Bestätigung | Team | N Abschn. 11.1 |
| 3 | `throwdet`-Fold-Modelle (OOF) auf ganzen Aufnahmen 02/03/04 testen. Gleiche Kennzahl wie bei Tims Modellen, also `tools/early_stream.py` bzw. `throwdet/replay.py`. **[Annahme]** Für Laurenz' Checkpoint-Format braucht `early_stream.py` evtl. einen Adapter. | offen (Noah hat das Werkzeug, Laurenz die Modelle) | N Abschn. 9.3 |
| 4 | Tims `early_cv`: Training auf durchgehenden 3–5-s-Fenstern mit echten Pausen, BN auf sauberen Daten, Dichte prüfen. Oder verwerfen zugunsten von `throwdet`. | Tim | N Abschn. 9.3 |
| 5 | Frische Kalibrier-Aufnahme im Demo-Aufbau, θ vorher festlegen; ggf. `adapt.sh` | offen | N Abschn. 11.2, L README „Before the demo“ |
| 6 | Latenz Sensor → Kommando messen | offen | N Abschn. 11.3, Track-Brief |
| 7 | Test am echten Go2: Sport-API-Reflex (`tools/go2_reflex.py`) bzw. Low-Level-Dodge (`go2_modes.py`, erst im Geschirr) | Noah (Sport-API), Tim (Low-Level) | `tools/go2_reflex.py`, T `sim/deploy/README.md` |
| 8 | Ausweich-Zeitbudget (W6): Hund beim „arm“ bereits in Bewegung? Beim Low-Level-Dodge die 2 s Traben vorziehen? | offen | W6 |
| 9 | ROS-Kette `ball_trigger_udp` → `two_stage_reflex` ist nicht in ROS gelaufen. Für `throwdet` gibt es keinen ROS-Knoten; Laurenz' `live_readout.py` muss an den Trigger angebunden werden. | Noah / Laurenz | N Abschn. 4.2, L README „Integration“ |
| 10 | Zweiter Seed für `w2_paste` fehlt; x1/x2 unbelegt (nur relevant, wenn der Ball-Strang in die Demo kommt) | Noah | N Abschn. 5, 11.6 |
| 11 | Alles nach `main` zusammenführen und committen, README, Durance-Material prüfen | Team (je Strang) | N Abschn. 11.5 |
| 12 | Write-ups von Tim und Laurenz in `docs/writeups/` ablegen. Tims `~/early` ist bisher undokumentiert. | Tim, Laurenz | diese Datei |

---

## 7. Was diese Zusammenfassung nicht abdeckt

- **Andere Agent-Sessions:** Auf diesem Rechner gibt es nur zwei T3-Threads: „Ultracode Handoff Status Review“ (Noah) und diesen. Sessions auf Tims oder Laurenz' Rechnern sehe ich nicht.
- **Panel-Talk der Veranstaltung:** Es gibt keinen Mitschnitt und keine Notizen. „Panel“ meint hier die Panel-Analysen der Agent-Sessions in `noah_sources/panel_*.md`.
- **Kleinere frühere Stränge vom 21.11. nachmittags**, ohne Write-up:
  - Noahs `training/train_throw_own.py` (`runs/own_throw`, A100 `~/noah_throw`);
  - der 3-Klassen-Lauf in `~/snn_pretrain/runs_throw`: CV-Genauigkeit 0,63–0,76 auf 38 Clips; nur auf Synth trainiert und auf echten Daten getestet 0,18–0,37. Urheber aus den Dateien nicht ersichtlich.
  - Beide sind durch die späteren Stränge überholt.
- **`~/extpos`** (A100, 22.11. ca. 08:00): Prüfung, ob sich DailyDVS- bzw. HMDB-Positive (v2e) von eigenen unterscheiden lassen.
  - DailyDVS-Positive gegen eigene Positive: 0,58–0,66 balanced accuracy. Nur schwach unterscheidbar.
  - HMDB gegen eigene: 0,94 beim Sensor-Merkmal. Leicht unterscheidbar, also eher ungeeignet.
  - Die HMDB-Auswertung ist mit einem Fehler abgebrochen (fehlende `labels.csv`). Kein Write-up, Urheber vermutlich Tim **[Annahme]**.
- **Beobachtung:** Der A100 war um 10:30 untätig (0 % GPU, keine Python-Jobs). Ich habe dort nichts gestartet oder gestoppt.

---

## 8. Quellenverzeichnis

Abkürzungen in den Tabellen: **N** = Noahs Write-up, **L** = Laurenz, **T** = Tim.

| Kürzel / Datei | Ort | Belegt |
|---|---|---|
| N: [`noah_ball_trigger.md`](noah_ball_trigger.md) | `docs/writeups/` | Strang A komplett; Bugs in `robust_cv.py`; FPGA-Stream-Test; Test von Tims Früherkennung auf ganzen Aufnahmen; Architektur und Latenz |
| [`noah_sources/w2_paste_summary.json`](noah_sources/w2_paste_summary.json) | `docs/writeups/noah_sources/` | Tabelle FPGA ganze Aufnahmen (θ-Sweep, UND/ODER mit Tracker) |
| [`noah_sources/r2_ext_fa_0203.json`](noah_sources/r2_ext_fa_0203.json), [`r2_old_fa_0203.json`](noah_sources/r2_old_fa_0203.json) | ebd. | Tims Modelle auf ganzen Aufnahmen |
| [`noah_sources/handoff_overnight_2026-11-21.md`](noah_sources/handoff_overnight_2026-11-21.md), `handoff.md` (Repo-Root, gleicher Inhalt) | ebd. | Stolperdraht, Tracker, Deploy-Risiken, Datenlage |
| `noah_sources/panel_0*.md`, `research_0*.md` | ebd. | Kennzahl-Wahl, Winner's curse, DemoCost; Literatur (Falanga, Forrai, Zhu, Delbruck, REBot) |
| L: `throwdet/README.md`, `throwdet/results_summary.md`, `throwdet/models/*/cv_readout.txt`, `readout.json` | Branch `origin/throw-detector` (Commit `c677655`, 22.11. 09:23); A100 `~/throwdet/` | Strang B: Daten, Ergebnisse aller 37 Läufe, Seed-Streuung, FPGA-Check, Firmware #43, Deploy/Backup, Zeitbudget |
| `dataset/README.md` | Branch `origin/data-pipeline` | Laurenz' Augmentierung, Synth v2, Video, Shortcut-Probes |
| A100 `~/ext/dailydvs/REPORT.md` | A100 | DailyDVS-Import: Realismus- und Trennbarkeits-Probes |
| T: `experiments/robust_release/README.md` | Branch `origin/Tim` | Strang D |
| T: `training/pretrain/README.md` | Branch `origin/Tim` | Strang E |
| T: `sim/README.md`, `sim/deploy/README.md`, `sim/avoid/README.md` | Branch `origin/Tim` | Strang F (MuJoCo-Dodge, Supervisor, Moduswechsel, Lidar-Wächter) |
| T: A100 `~/early/runs_ext/summary.json`, `~/early/runs_early/summary.json`, `launch.sh`, `jobs_e1.txt` | A100 (lokale Kopie `runs/early_tim/`) | Strang C |
| `runs/two_stage/all_runs.log` | Repo (git-ignoriert) | Gazebo-Zeitmessung `two_stage_reflex` |
| `tools/go2_reflex.py` (Docstring) | Repo, nicht committet | Sport-API-Reflex am echten Go2, Bedienung |
| A100 `~/extpos/probe_*.json`, `~/snn_pretrain/runs_throw/summary.json` | A100 | Abschnitt 7 |
| `docs/TRACK_NOTES.md` | Repo | Jury-Gewichtung, Abgabe-Anforderungen |
| T3-Thread „Ultracode Handoff Status Review“ | T3 | Auftrag zum Label-Wechsel (22.11. ca. 10:20), x1_trackaug-Zwischenstand |
