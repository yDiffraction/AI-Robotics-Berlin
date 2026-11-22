# Ball-Ausweich-Trigger mit Event-Kamera und SNN auf dem FPGA – Write-up (Team Noah)

Durance Vision Lab @ Berlin HackLab Robotics, 21.–22.11.2026. Stand: 22.11., ca. 11 Uhr.
Event-Kamera (Prophesee GenX320) → Spiking-CNN auf dem Kria KV260 (RiverFPGA) → Host-Trigger → ROS 2 → Unitree Go2.

Alle Zahlen stammen aus eigenen Läufen. **Gemessenes** und **Annahmen** sind getrennt markiert. Pfade beziehen sich auf das Projekt-Repo.

---

## 1. Kurzfassung

1. **Ziel:** Der Go2 soll einem geworfenen Ball ausweichen, aber **nicht** bei Täuschungen (Wurfbewegung ohne Ball) und **nicht** in Ruhe.
2. **Zentrale Erkenntnis:** Täuschungen unterscheiden sich von Würfen nur durch den **Ball im Luftraum über dem Werfer**, nicht durch die Armbewegung.
   - Tims Ausgangs-Setup lag bei Wurf vs. Täuschung bei AUC 0,53–0,59, also Zufallsniveau.
   - Ein Großteil davon waren **Fehler im Trainingscode**. Unter anderem lag das Ziel zeitlich falsch, teils bis zu 200 ms *vor* dem Abwurf (Abschnitt 4).
3. **Bestes Modell `w2_paste`:** Tims v1-Architektur, volles Bild. Im Training werden echte Ballflugbahnen in Täuschungs-Clips eingesetzt und als „Wurf“ gelabelt.
   - Within-fold-AUC Wurf vs. Täuschung **0,991**, gepoolt 0,965.
   - **Ball-Test:** Wird der Ball aus einem Wurf entfernt, löst das Netz nie aus (0 %). Wird er in eine Täuschung eingesetzt, löst es in 76 % der Fälle aus. Das Netz entscheidet also wirklich am Ball.
4. **Auf dem echten FPGA mit ganzen, zurückgehaltenen Aufnahmen** (keine geschnittenen Clips), θ=5:
   - **42/42 Würfe** erkannt;
   - **8/65 Täuschungen** lösen aus;
   - **1,7 Fehlauslöser/min** in den Pausen.
   - Kombiniert per UND mit einem klassischen Ball-Tracker: 34/42 Würfe, **0/65 Täuschungen, 0,1–0,2/min**.
5. **Eingabepfad:** Der Laptop baut aus den Kamera-Events selbst 0/1-Bilder (so wurde trainiert) und schickt sie an den SPLEAT-Server.
   - 100 Bilder/s statt 47/s über den Board-eigenen Pfad.
   - Modell-Latenz 0,45–0,6 ms, Round-Trip etwa 2 ms.
   - FPGA und Fixpunkt-Simulation stimmen überein.
6. **Deployt und live gezeigt:** Das Modell läuft auf dem Board. Die Live-Anzeige ist in die Kamera-Webseite eingebaut und zeigt Evidenz, Schwelle, Spikes und ein „AUSWEICHEN!“-Overlay.
7. **Neue Aufgabe seit 22.11. vormittags:** Wurf und Täuschung bekommen **dasselbe Label** („Wurfbewegung findet statt“), möglichst früh erkannt. Dazu gibt es Tims Früherkennungs-Modell.
   - Auf ganzen echten Aufnahmen ist es deutlich schwächer als in seiner Auswertung auf zusammengesetzten Clips.
   - Analyse und Verbesserungsvorschläge in Abschnitt 9.

---

## 2. Aufgabe, Bewertung, Hardware

**Jury-Gewichtung:**

| Kriterium | Gewicht |
|---|---|
| Working System | 35 |
| Robustness & control | 25 |
| Originality | 20 |
| Engineering | 10 |
| Reproducibility | 10 |

**Levels:**
- **Level 2:** eigene Daten aufnehmen, eigenes SNN trainieren, auf das FPGA bringen, ROS-2-Verhalten auf dem Go2.
- **Level 3:** robust gegen ungesehene Aufnahmen und Distraktoren, mit *messbarer* Verbesserung.

**Abgabe:**
- 2-Minuten-Demo.
- Zahlen: Erfolge/Versuche, False Triggers und Latenz vom Sensor bis zum Kommando, getrennt von der reinen Inferenzzeit.
- Repo mit README, Training und Launch. Durance-Tools und -Datensatz nicht veröffentlichen.

**Pipeline auf dem Kria:** `despot` (Events → Bilder, 100 Hz, Downsampling 4 → 80×80) → `spleat-server` (TCP 3333, CBOR) → RIVER IP im FPGA.

**Firmware-Randbedingung:**
- Auf dem Board läuft eine arch-v1-Firmware mit **genau 11 Ausgängen** (Tims DVS-Gesture-Modell).
- Ein neues Modell lässt sich ohne neuen Bitstream nur deployen, wenn der Graph identisch ist. Dann reicht der Tausch der `weights.txt`.
- Architektur v1:
  - 4 Spiking-Conv-Schichten mit 16/32/64/64 Kanälen;
  - Stride 2, Kernel 5/3/3/3;
  - Conv-Kopf, keine vollverbundenen Schichten;
  - LIF-Neuronen;
  - 78501 Werte.
- Fixpunkt: Gewichte und Bias Q2.6, Potentiale, Schwelle und Leak Q4.12.

---

## 3. Daten

| Satz | Inhalt | Ball nach dem Abwurf sichtbar? |
|---|---|---|
| 01 | wenige Würfe und Täuschungen, Person zu nah | nein, der Ball trennt sich nicht vom Arm |
| 02 | 5 Würfe, 10 Täuschungen, zweite Person läuft vorbei | ja |
| 03 (6 Aufnahmen) | 37 Würfe, 55 Täuschungen; 3 Aufnahmen enthalten nur Würfe | ja |
| 04 Wurf-Aufnahme | 28 Würfe, 36 Täuschungen | **nein**, der Kopf ist am oberen Bildrand; „Wurf“ heißt hier nur Armbewegung |
| 04 Ruhe-Aufnahme | lange, unruhige Ruhe-Szene (Decke, Bewegung im ganzen Bild) | – |

- **Labels pro Aufnahme:** `label` (throw, feint, idle, invalid) und `decision_s` (Schwung-Höhepunkt).
- **Trainings-Clips:** 1,2 s lang, Anker bei 700 ms = Schwung-Höhepunkt.
- **Echter Abwurf-Zeitpunkt (`release_ms`):** per Ball-Tracker automatisch gesetzt, für 33 der 42 Würfe aus 02/03 (610–870 ms).
- **Folds** (nach Aufnahme-Session gruppiert, damit Test-Sessions ungesehen bleiben):
  - A = 2 Aufnahmen aus 03;
  - B = 3 Aufnahmen aus 03;
  - C = 02 + 1 Aufnahme aus 03 + 01;
  - D = 04-Wurf-Aufnahme (nur Test).
- **Bekannte Schwächen:**
  - Nur etwa 42 Würfe mit sichtbarem Ball.
  - Die Ruhe-Clips stammen größtenteils aus einer anderen, unruhigen Szene. Das lädt zur Abkürzung „volles Bild = Ruhe“ ein.
  - Allein der Zufalls-Seed verschiebt die AUC um ±0,03–0,07.

---

## 4. Vorgehen

### 4.1 Analyse-Panel und Fehler im Ausgangs-Code

Ein Agent-Panel mit 5 Blickwinkeln (Labels, Readout, Architektur/FPGA, synthetische Daten, Eval-Skeptiker) hat den Ausgangszustand untersucht und einen Experimentplan aufgestellt (Anhang A).

**Gefundene Fehler in Tims Trainingscode** (`robust_cv.py`). Alle sind in unserer Pipeline `ball_cv.py` behoben:
1. **Zeitliches Verschieben und Strecken:** Beim Verschieben (±8 Bilder) und Strecken der Fenster bleibt das Ziel fest bei Frame 70. Es liegt dadurch bis zu ±80 ms falsch, bei 02 bis zu 200 ms vor dem Abwurf. Das Netz lernt so, schon auf das Ausholen zu reagieren, und genau das imitieren Täuschungen.
2. **`release_ms` ignoriert:** Die Clip-Aufbereitung wertet den echten Abwurf-Zeitpunkt nicht aus.
3. **Zeitachse der Ruhe-Clips:** Gezählt wird ab dem ersten Event statt ab dem Clip-Start. Ruhe-Clips sind dadurch um bis zu 222 ms verschoben.
4. **Rückwärts abgespielte Ruhe-Fenster:** Sie laufen im Ping-Pong teils rückwärts. Das ist ein unphysikalisches Signal, das es nur im Training gibt.
5. **BatchNorm-Statistiken:** Sie werden auf augmentierten, also ausgedünnten Daten berechnet.
6. **Loss-Mittelung und Symmetrie-Term:**
   - Die kumulative Mittelung ab Fensterstart verwässert den kurzen Ball-Impuls (5–10 Bilder).
   - Der Symmetrie-Term erzwingt Gleichheit von Wurf und Täuschung genau in den Bildern, in denen der Ball sichtbar ist.
7. **Vertikale Verschiebung:** 8 Zellen schieben den Ball aus dem Bild.

### 4.2 Klassische Baselines (kein Training, CPU)

- **„Stolperdraht“:** aktive Zellen in den obersten 8 Bildzeilen, summiert über 2 Bilder. Auf 02/03 erkennt er 42/42 Würfe bei 0/65 Täuschungen. Diese Zahl ist **in-sample**.
- **Kausaler Ball-Tracker:**
  - **Funktionsweise:** Er sucht kleine, isolierte Blobs und verkettet sie über Bilder. Er löst aus, wenn eine Kette gleichmäßig nach oben steigt. Geprüft werden dabei:
    - der Konstant-Geschwindigkeits-Fit;
    - die Residuen-Grenze;
    - die „Neuheit“ der Pixel;
    - der Abstand zum Körper.
  - **Ergebnis:** 34/42 Würfe, im Median 40 ms nach dem Abwurf, **0/104 Täuschungen über alle Datensätze**. Ruhe: 0,2–0,8 Fehlauslöser pro Minute.
  - **Vorbehalt:** Die Parameter wurden auf diesen Daten eingestellt, die Zahlen sind also optimistisch.
- **Konsequenz:** Die Information steckt im Luftraum. Das SNN muss lernen, auf den Ball zu achten.

### 4.3 SNN-Training: was das Netz auf den Ball lenkt

Die Pipeline `ball_cv.py` baut auf Tims Code auf: gleiche Architektur, Vortraining auf DVS Gesture, gleiche Augmentation.

**Änderungen:**
- **Fenster-Loss:** Cross-Entropy auf der Spike-Summe der letzten 10 Bilder, genau die Größe, die live auslöst.
  - Ziel „Wurf“: 3–23 Bilder nach dem Abwurf.
  - „Kein Wurf“: auf dem Ausholen vor dem Abwurf, auf allen Täuschungs- und allen Ruhe-Bildern.
- **Ball einsetzen (`--synth-throw`):** Echte Ballflugbahnen (vom Tracker extrahiert, nur aus Trainings-Sessions) werden in Täuschungs-Clips eingesetzt, genau in deren Schwung-Phase. Diese Clips sind als „Wurf“ gelabelt. Damit ist der Ball der **einzige** Hinweis, der zu allen Labels passt.
- **Weitere Optionen,** jeweils einzeln getestet:
  - Ball aus Würfen entfernen (`--erase-share`);
  - Hilfskopf, der die Ball-Zellen vorhersagt (nur im Training);
  - MIL-Loss;
  - Ausschnitt nur der oberen Zeilen (ROI);
  - Würfe ohne sichtbaren Ball weglassen;
  - Ruhe aus derselben Szene;
  - dritter Ausgang „Ausholen“ für einen zweistufigen Reflex;
  - Variation der Ballspuren (Tempo ±30 %, Bogen ±25 %, Starthöhe);
  - stärkere Überlagerung mit der unruhigen Ruhe-Szene.
- **Pro Lauf:** 1200 Schritte, 3 Folds. Jeder Lauf dauert etwa 25 min auf einer RTX 4070, auf dem A100 laufen 2 parallel.

**Auswertung:** Ganze zurückgehaltene Clips werden kausal abgespielt.
- Evidenz S(t) = Summe über 10 Bilder von (Wurf-Spikes − Kein-Wurf-Spikes).
- Clip-Score für Würfe: max S im Fenster [Abwurf, Abwurf + 250 ms].
- Clip-Score für Täuschungen und Ruhe: max S über den ganzen Clip.
- **Hauptkennzahl (Panel):** within-fold AUC Wurf vs. Täuschung auf 02/03.
- **Entscheidungsregel:** Eine Änderung zählt nur, wenn sie um mindestens max(0,03; Seed-Rauschen) besser ist und in mindestens 2 von 3 Folds gewinnt.

---

## 5. Ergebnisse der Kreuzvalidierung (Clips, Float-Modell)

582 gemeinsame zurückgehaltene Clips: 49 Würfe, 68 Täuschungen, 465 Ruhe.

**Spalten:**
- `prim`: Hauptkennzahl (within-fold AUC Wurf vs. Täuschung auf 02/03).
- `auc_tf`: gepoolte AUC Wurf vs. Täuschung.
- `rec250`: Anteil der Würfe, die ≤ 250 ms nach dem Abwurf auslösen, bei einer Schwelle mit 10 % Täuschungs-Fehlauslösern.
- `busy%`: Anteil der unruhigen Ruhe-Clips, die auslösen.
- `Ball eingesetzt`: löst aus, wenn ein Ball in eine Täuschung eingesetzt wird (hoch = gut).
- `Ball entfernt`: löst noch aus, wenn der Ball aus einem Wurf entfernt wird (niedrig = gut).

| Lauf | Änderung | prim | auc_tf | rec250 | busy% | Ball eingesetzt | Ball entfernt |
|---|---|---|---|---|---|---|---|
| b0_tim | Ausgangs-Setup (mit Fehlern), Folds A+B | ~0,56 | 0,59 / 0,53 | – | – | 22–30 % | 0 % |
| w1_win | Fehler behoben, Fenster-Loss | 0,944 | 0,872 | 0,63 | 7 % | 68 % | 0 % |
| w1_win_s1 | anderer Seed | 0,978 | 0,938 | 0,49 | 8 % | 57 % | 0 % |
| **w2_paste** | + Ball in Täuschungen eingesetzt | **0,991** | **0,965** | **0,76** | 10 % | 76 % | 0 % |
| f1_ball | + Einsetzen + Entfernen + Hilfskopf | 0,960 | 0,874 | 0,61 | 6 % | 77 % | 0 % |
| f1_ball_s1 | anderer Seed | 0,991 | 0,941 | 0,82 | 9 % | 85 % | 0 % |
| a1_aux | Hilfskopf für Ball-Zellen | 0,964 | 0,866 | 0,59 | 8 % | 67 % | 3 % |
| w4_erase | Ball aus Würfen entfernt | 0,928 | 0,791 | 0,12 | 3 % | 19 % | 0 % |
| m1_mil | MIL-Loss | 0,939 | 0,824 | 0,69 | 2 % | 75 % | 6 % |
| d1_drop | Würfe ohne sichtbaren Ball weggelassen | 0,931 | 0,801 | 0,65 | 0 % | 68 % | 12 % |
| r1_roi16 | Netz sieht nur die oberen 16 Zeilen | 0,951 | 0,660 | 0,29 | 3 % | 25 % | 0 % |
| r2_roi24 | nur die oberen 24 Zeilen | 0,988 | 0,795 | 0,76 | 3 % | 80 % | 9 % |
| x1_trackaug | w2_paste + Spur-Variation | – | 0,950 | (rec@10 % 0,84) | 10 % | 86 % | 0 % |
| x2_trackaug_ov | x1 + stärkere Überlagerung | – | 0,896 | (rec@10 % 0,71) | **1,4 %** | 81 % | 0 % |

**Lesart:**
- **Die Fehlerbehebung allein** hebt die Hauptkennzahl von etwa 0,56 auf 0,94–0,98. Das Seed-Rauschen liegt bei etwa 0,03.
- **Ball einsetzen (`w2_paste`)** ist die einzige Änderung, die die Entscheidungsregel klar besteht: +0,047, Bootstrap-CI [+0,012; +0,094], besser in allen 3 Folds.
- **Ausschnitt nur der oberen Zeilen (ROI)** ist nicht besser als das volle Bild und als Demo weniger überzeugend. Deshalb fiel die Wahl auf das volle Bild.
- **Stärkere Überlagerung (`x2`)** senkt die Fehlauslöser in der unruhigen Szene deutlich, kostet aber Trennschärfe gegen Täuschungen. Das ist nur 1 Seed, also nicht belastbar.
- **Ball entfernen (`w4_erase`)** schadet. Annahme: Das Netz wird insgesamt zu vorsichtig.

---

## 6. Ergebnisse auf dem echten FPGA mit ganzen Aufnahmen

**Aufbau:**
- Pro Fold wird das Fold-Modell auf das Board gebracht. Dann laufen die **ganzen** Aufnahmen dieses Folds durch: Laptop-0/1-Bilder → FPGA → Auslöse-Logik.
- Evidenz d = Summe über 10 Bilder von (Wurf- − Kein-Wurf-Spikes). Auslösen bei d ≥ θ, danach 2 s Pause.
- Treffer: Auslösen innerhalb von [Label − 0,4 s; Label + 0,8 s].
- Umfang: 42 Würfe, 65 Täuschungen, 8,8 min Pausen.

| Regel | Würfe | Täuschungen lösen aus | Ruhe-Fehlauslöser |
|---|---|---|---|
| SNN θ=3 | 42/42 | 16/65 | 3,95/min |
| SNN θ=4 | 42/42 | 12/65 | 2,15/min |
| **SNN θ=5 (Live-Standard)** | **42/42** | **8/65** | **1,69/min** |
| SNN θ=6 | 38/42 | 6/65 | 1,36/min |
| SNN θ=8 | 34/42 | 3/65 | 0,45/min |
| nur Ball-Tracker (Host) | 33/42 | 0/65 | 0,56/min |
| **SNN UND Tracker** (beide innerhalb von 150 ms), θ=3–5 | 34/42 | **0/65** | **0,11–0,23/min** |
| SNN ODER Tracker, θ=5 | 41/42 | 8/65 | 2,15/min |

**Hardware-Zahlen (gemessen):**
- **FPGA vs. Fixpunkt-Simulation**, 60 Clips, 9000 Bilder: Spikes 58 vs. 61 bzw. 1891 vs. 1905. Entscheidungen pro Clip stimmen in 95–100 % überein.
- **Float vs. Fixpunkt:** Wurf-Spikes 323 → 264, 91 % der Einträge gleich. Schwellen deshalb **auf dem Board** kalibrieren, nicht am Float-Modell.
- **Latenz und Durchsatz:**
  - Modell-Latenz auf dem Board: Median 0,5–0,6 ms, p95 3,6 ms.
  - Round-Trip Laptop ↔ Board: etwa 1,9 ms.
  - 100 Bilder/s bei 100 Hz Takt, offline bis 480 Bilder/s.
- **Live-Test** mit Echtzeit-Wiedergabe einer Aufnahme als falsche Kamera: 3/3 Würfe, höchstens 50 ms nach dem Label, 1 Fehlauslöser in Ruhe, 0 Täuschungen.

**Vorbehalte:**
- Die Schwellen wurden auf denselben zurückgehaltenen Daten gewählt, die Zahlen sind also leicht optimistisch.
- Die **Gesamtlatenz vom Sensor bis zum Kommando ist noch nicht gemessen.**

---

## 7. System-Architektur

```
GenX320 ─MIPI→ Kria (despot_debug_camera) ─TCP, Events→ Laptop
  Laptop: Events → binäre 2×80×80-Bilder pro 10 ms (identisch zur Trainingsdaten-Erzeugung)
          ─TCP/CBOR→ Kria spleat-server → FPGA (v1, 11 Ausgänge: ch0 Wurf, ch1 kein Wurf, Rest = 0)
          ←─ Spikes pro Bild
          Auslöse-Logik (54 Unit-Tests):
             d = Σ10 (Wurf − kein Wurf) ≥ θ → „arm“ + „dash“, danach 2 s Refraktärzeit
             Watchdog: > 200 ms keine Bilder → Reset; nie Ausweichen auf veralteten Daten
             ungültiges Bild (NaN, Wert ≠ 0/1, Spike auf Füll-Kanal = falsches Modell geladen) → Reset
          optional: klassischer Ball-Tracker als UND-Bestätigung
          ─UDP/JSON→ ROS 2: ball_trigger_udp → /reflex/arm|dash|disarm → two_stage_reflex → Sport-API → Go2
```

**Warum der Laptop die Bilder baut:**
- Der Board-eigene Pfad schickt **Zählwerte**, trainiert wurde aber auf 0/1. Ein älteres Modell ist daran auf dem Board kollabiert.
- Außerdem kamen über diesen Pfad nur etwa 47 statt 100 Bilder/s an, und das in Schüben.

**Alternative:** ein kleiner Proxy auf dem Kria, der Zählwerte in 0/1 umwandelt. Er ist geschrieben, aber **nicht getestet**.

**`two_stage_reflex`:**
- Bei „arm“ einen Schritt nach vorn, bei „dash“ mit 1 m/s zurück, danach zurück zur Startposition.
- Bewegung nur bei freiem Lidar-Status, jede Bewegung mit Zeitlimit.
- In der Simulation (Gazebo/MuJoCo) gebaut. Die UDP→ROS-Brücke ist **noch nicht in ROS getestet**.

**Export:**
- Der Kopf wird mit Nullen auf 11 Ausgänge aufgefüllt; die Füll-Kanäle feuern nie.
- Ein Regressionstest reproduziert Tims Board-Gewichte bitgenau.
- Fixpunkt-Check mit 0/1- und Zählwert-Eingabe.

---

## 8. Recherche: Kernaussagen

Literatur mit Links in Anhang B.

- **Ausweichen und Fangen in der Praxis:** Alle Systeme erkennen das **Objekt** direkt und prüfen die Physik.
  - Falanga 2020: Drohne, 3,5 ms Wahrnehmung.
  - Forrai 2023: ANYmal fängt, RANSAC-Parabel.
  - Zhu 2023: Mini-Cheetah weicht aus.
  - Delbruck 2013: Roboter-Torwart, 2,2 ms.
  - EV-Catcher 2022: binäre Event-History-Bilder.
  - Überwiegend klassisch, ohne Training. Das stützt die Kombination aus SNN und Tracker.
- **Zeitbudget Go2 (REBot 2025):** Unter 0,5 s Vorwarnung gelingt ein Ausweichen fast nie (5 %). Bei 0,5–1,5 s sind es 65 %. Bei einem hohen Bogen muss der Trigger also innerhalb von etwa 50–100 ms nach dem Abwurf kommen.
- **Originalität:** Vorarbeit zur Erkennung des **Abwurf-Moments** mit Event-Kamera haben wir nicht gefunden.
- **SNN-Readout und Loss:**
  - Fenster-, EMA- oder MIL-Loss nach dem Vorbild von Streaming-Keyword-Spotting (Amazon, Snips, SynSense Xylo).
  - Die Schwelle bei fester Fehlalarm-Rate wählen.
  - CUSUM/SPRT als Alternative zur Fenstersumme.
- **Augmentation:**
  - Objekt-Einsetzen (copy-paste) ist bei wenigen Positiven der stärkste Hebel. Das ist bei uns bestätigt (`w2_paste`).
  - Kontrafaktische Paare (Ball rein/raus) erlauben eine gezielte Prüfung, ob das Netz am richtigen Merkmal entscheidet.
  - Prozedurale Ballspuren (ShapeAug, EVDodgeNet).
- **Biologisch inspiriert:** OMS-Retina, STMD- und Looming-Detektoren, TDE-Bewegungsdetektoren. Sie unterdrücken große Körperbewegung und heben kleine, neu auftauchende Objekte hervor. Das ist die Grundlage für das Isolations- und Neuheits-Gate im Tracker und eine mögliche fest verdrahtete erste SNN-Schicht.
- **Sensor:** Die GenX320 hat On-Chip-Filter (ERC, STC). Bei viel Bewegung kann das ERC Events kappen, das ist eine mögliche Ursache für den unsichtbaren Ball in 04 (Annahme).

**Demo-Ideen:**
- **Zweistufiger Reflex:** beim Ausholen bereitmachen, beim Ball ausweichen, bei einer Täuschung sichtbar entspannen.
- **Folie zum Ball-Test:** Ball entfernen → kein Auslösen; Ball einsetzen → Auslösen.

---

## 9. Neue Aufgabe: Früherkennung „Wurfbewegung“ (Wurf = Täuschung = positiv)

### 9.1 Tims Ansatz

- **Ein Ausgangsneuron:** Der Host bildet eine EMA (α=0,85 pro 10 ms) und vergleicht mit einer Schwelle θ. Ziel ist 1 ab 300–500 ms vor dem Schwung-Höhepunkt bis 300 ms danach.
- **Daten:**
  - Positive: eigene Würfe und Täuschungen.
  - Negative: eigene Ruhe, motion_buddy (Durance), DailyDVS-200, DVS Gesture.
  - Zurückgehaltener Fremdkamera-Test: DailyDVS-Wurf/Toss.
- **Training auf zusammengeklebten 3-s-Streams** aus 1-s-Clips verschiedener Quellen. Gegen Quellen-Abkürzungen:
  - Dichte-Normalisierung;
  - eine schwache Schicht fremder Negative unter dem ganzen Stream;
  - Polaritätstausch, Spiegeln, Verschieben, Zoom, Zeitstreckung;
  - Rauschen und Hot Pixels.

  Leak-Zeitkonstanten ≤ 8 Bilder.
- **Seine Auswertung** (auf zusammengesetzten Streams):

  | θ | Würfe | Täuschungen | Latenz vor Höhepunkt (Wurf / Täuschung) | Fehlalarme eigene Szene |
  |---|---|---|---|---|
  | 0,80–0,95 | 52–56 % | 57–63 % | 310–345 / 410 ms | 1,5–2/min |

### 9.2 Unser Test: Tims Fold-Modelle auf ganzen echten Aufnahmen

**Aufbau:**
- Jede Aufnahme läuft mit dem Fold-Modell, das sie **nicht** gesehen hat.
- 0/1-Bilder wie live, Float-Modell.
- Treffer: erster Alarm in [−1,0 s; +0,3 s] um den Schwung-Höhepunkt.
- Fehlalarm: Alarm außerhalb von ±1,5 s um jedes gelabelte Ereignis.
- In 02/03 liegen die Ereignisse dicht, es bleibt nur etwa 1 min Pausen-Zeit. Die Fehlalarm-Raten sind deshalb grob.

| Modell, θ | Würfe 02/03 | Täuschungen 02/03 | Latenz-Median (Wurf / Täuschung) | Fehlalarme in Pausen |
|---|---|---|---|---|
| r2_old_fa, 0,80 | 25/42 | 43/65 | −260 / −550 ms | 56/min |
| r2_old_fa, 0,95 | 11/42 | 17/65 | −10 / −420 ms | 21/min |
| r2_ext_fa, 0,80 | 14/42 | 19/65 | −65 / −410 ms | 22/min |
| r2_ext_fa, 0,95 | 8/42 | 14/65 | −120 / −25 ms | 4/min |
| r2_ext_fa, 0,80, 04-Wurf-Aufnahme | 10/29 | 17/36 | −560 / −170 ms | 10/min |

**Deutung** (teils Annahme, jeweils mit Begründung):
1. **Die Auswertung auf zusammengeklebten Streams ist zu optimistisch.** Dort beginnt jeder Positiv-Clip 700 ms vor dem Höhepunkt mit einem harten Wechsel von fremden Daten zur eigenen Szene. Reagiert das Modell auf diesen Wechsel, zählt das als „früh“. In echten Aufnahmen gibt es keinen Wechsel.
2. **Die echten Pausen fehlen im Training.** Zwischen den Würfen läuft die Person herum, hebt den Ball auf und geht zurück. Das sind die realistischen harten Negative, und dort feuert das Modell oft.
3. **Dichte-Normalisierung:** Im Training wird auf 80–500 Events pro Bild ausgedünnt, live nicht. Ist die eigene Szene dichter, sieht das Netz live eine nie trainierte Verteilung. **Nicht geprüft.**
4. **BatchNorm-Statistiken** werden auf augmentierten Daten berechnet, derselbe Fehler wie in 4.1 Punkt 5.

### 9.3 Verbesserungsvorschläge

1. **Auswertung auf ganzen Aufnahmen** als Hauptkennzahl. Das Werkzeug ist fertig (`tools/early_stream.py`). Für belastbare Fehlalarm-Raten braucht es zusätzlich 5–10 min Pausen-Aufnahme im Demo-Aufbau: Herumlaufen, Ball aufheben, Winken.
2. **Training auf durchgehenden 3–5-s-Fenstern aus den eigenen Aufnahmen**, zusätzlich zu den zusammengeklebten Streams:
   - Ziel um **jeden** gelabelten Wurf und jede Täuschung;
   - alles dazwischen ist negativ;
   - um „invalid“ wird ignoriert.

   Das liefert die realistischen harten Negativen und entfernt die Szenenwechsel.
3. **BatchNorm auf unaugmentierten Daten auffrischen.**
4. **Event-Dichte prüfen:** eigene Aufnahmen gegen den Trainingsbereich vergleichen. Notfalls anpassen oder live genauso ausdünnen.
5. **Zweistufige Demo:**
   - Früherkennung → „arm“ (bereitmachen);
   - Ball-Modell oder Tracker → „dash“ (ausweichen).

   Zwei Netze passen nicht gleichzeitig auf die Firmware. Ein gemeinsames Netz mit drei Ausgängen (Ausholen / Wurf / kein Wurf) ist möglich: Ein Testlauf mit drittem Ausgang „Ausholen“ erreichte für Ausholen vs. Ruhe AUC etwa 0,82 (2 Folds).
6. **Kostenfunktion vorher festlegen**, damit θ nicht nachträglich auf Testdaten optimiert wird. Panel-Vorschlag: 10·Fehlrate + 10·Täuschungs-Fehlauslöser + 5·Ruhe-Fehlauslöser/min.

---

## 10. Offene Punkte

1. **Demo-Entscheidung zum neuen Label:** nur Früherkennung oder zweistufig (Früherkennung = bereitmachen, Ball = ausweichen)?
2. **Frische Kalibrier-Aufnahme im Demo-Aufbau:** statische Kamera, Ball sichtbar, Werfer etwa 3 m entfernt, freie Fläche über dem Kopf. 20 Würfe, 20 Täuschungen, 5–10 min Pausen-Bewegung. Danach die Schwelle festlegen.

   Das ist zugleich der einzige unverzerrte Test. Alle bisherigen Zahlen sind auf denselben Folds ausgewählt („winner’s curse“).
3. **Latenz Sensor → Kommando messen**, mit LED im Bild oder 240-fps-Video.
4. **ROS-Kette mit dem echten Go2 testen.** Die UDP→ROS-Brücke und `two_stage_reflex` sind bisher nur in der Simulation gelaufen.
5. **Reproduzierbarkeit:**
   - Trainings-Skripte aus `runs/` (git-ignoriert) nach `training/` und `tools/` übernehmen.
   - README und Commit fehlen noch.
6. **Statistik:** Für `w2_paste` fehlt ein zweiter Seed. Ob die Spur-Variation (`x1`, `x2`) wirklich hilft, ist offen.

---

## 11. Was ist neu, was übernommen

| Neu von uns | Von Tim | Starter / Durance |
|---|---|---|
| Trainings-Pipeline mit Fehlerbehebung und Ball-Einsetzen/-Entfernen, Bank- und Spur-Aufbereitung, Abwurf-Markierung, kausaler Ball-Tracker; Export mit Auffüllen auf 11 Ausgänge; Board- und Stream-Tests; Auslöse-Logik mit Tests; Live-Anzeige; UDP→ROS-Brücke; zweistufiger Reflex in der Sim | SNN-Architektur, Vortraining auf DVS Gesture, Basis-Trainingscode, Früherkennungs-Pipeline | Gesten-Knoten, Action-Bridge, Hindernis-Erkennung, Firmware, despot (nicht veröffentlichen) |

---

## 12. Dateien und Befehle

| Was | Wo / Befehl |
|---|---|
| Training Ball-Modell | `runs/ball/ball_cv.py` (Daten-Bank `make_bank.py`, Ballspuren `ball_tracks.py`) |
| Vergleich der Läufe | `cd runs/ball && python compare.py --base w1_win` |
| Export für das Board | `CUDA_VISIBLE_DEVICES= python training/export_ball.py --ckpt <final.pt> --out <dir> --check 60` |
| Deploy | `scp <dir>/weights.txt root@<kria>:/root/weights.txt && ssh root@<kria> systemctl restart spleat-server` |
| Board vs. Fixpunkt | `python tools/board_check.py --ckpt <fold.pt> --export <dir> --n 60` |
| Ganze Aufnahmen über das Board | `tools/stream_eval.sh <tag>` und `python tools/stream_summary.py <tag>` |
| Live mit Kamera | `python tools/record_tool.py` (Webseite mit SNN-Panel) oder `python tools/ball_live.py --theta 5 [--fire and] [--udp <host>:5555]` |
| ROS-Brücke | `ros2 run go2_sim_tools ball_trigger_udp` → `two_stage_reflex` |
| Früherkennung auf ganzen Aufnahmen | `python tools/early_stream.py --ckpt <model.pt> --rec <aufnahmen.bin> --out <dir>`, dann `--score '<dir>/*.npz'` |
| Auslöse-Logik testen | `python3 tests/test_ball_trigger_logic.py` (54 Tests) |

---

## Anhang A: Experimentplan des Panels (E1–E10) und was daraus wurde

| ID | Experiment | Ergebnis |
|---|---|---|
| E1 | Klassischer Tracker und reine Zähl-Baselines als Messlatte | Tracker 34/42 Würfe, 0/104 Täuschungen; Zählwerte im oberen Band AUC 1,00 in 02/03, über das ganze Bild nur 0,54 |
| E2 | Bereinigte Referenz `w1_win` mit 2 Seeds (Rauschgrenze) | 0,944 / 0,978 → Seed-Rauschen ≈ 0,034 |
| E3 | Deploy-Pfad prüfen: Fixpunkt, Zählwert-Eingabe, Ruhe-Stream | Fixpunkt ≈ FPGA; Zählwerte umgangen (Laptop baut 0/1-Bilder); Stream-Test auf dem Board gemacht |
| E4 | Würfe ohne sichtbaren Ball weglassen (`d1_drop`) | 0,931, nicht besser |
| E5 | Ball nur in Täuschungen einsetzen (`w2_paste`) | **0,991, übernommen** |
| E6 | Hilfskopf für Ball-Zellen (`a1_aux`) | 0,964, nicht signifikant |
| E7 | MIL-Loss (`m1_mil`) | 0,939, nicht besser |
| E8 | Ruhe aus derselben Szene im Training (`w3_sceneidle`) | nur teilweise gelaufen |
| E9 | Bündel der übernommenen Änderungen | entfällt, nur E5 übernommen → `final_w2_paste` |
| E10 | Host-Regel und Schwelle aus den Spikes, finales Modell | Fenster-Regel d ≥ θ mit Refraktärzeit; θ=5; finales Modell deployt |

**Weitere Warnungen des Panels:**
- Die gepoolte AUC ist zu etwa 87 % Wurf vs. Ruhe und mischt Fold-Modelle mit unterschiedlicher Kalibrierung. Deshalb nicht danach ranken.
- Seeds brauchen eigene Tags.
- Mit nur 1 Seed pro Variante sind Unterschiede erst ab etwa 0,10 nachweisbar.

---

## Anhang B: Literatur (Web-Recherche der Agents)

| Thema | Arbeit | Kernzahl | Bezug zu uns |
|---|---|---|---|
| Ausweichen | Falanga et al., Science Robotics 2020, https://doi.org/10.1126/scirobotics.aaz9712 | 3,5 ms Wahrnehmung, > 90 % Ausweichen bei 10 m/s | statische Kamera: jedes Event ist Bewegung; Neuheits-Karte |
| Fangen | Forrai et al., ICRA 2023 (ANYmal), https://arxiv.org/abs/2303.17479 | 83 % bis 15 m/s, klassische Erkennung | Physik-Check (Konstant-Geschwindigkeit/Parabel) gegen Fehlauslöser |
| Ausweichen | Zhu et al., IROS-Workshop 2023 (Mini-Cheetah), https://ippc-iros23.github.io/papers/zhu.pdf | 83 % in 24 Versuchen | Hysterese-Schwelle; früh entscheiden |
| Torwart | Delbruck & Lang 2013, https://www.frontiersin.org/articles/10.3389/fnins.2013.00223/full | 2,2 ms, 80 % abgewehrt | grobe Auflösung reicht; „Scharf schalten“ nur nach Ausholen |
| Fangen | EV-Catcher, RA-L 2022, https://arxiv.org/abs/2304.07200 | 81 % bei 13 m/s, 235 echte Beispiele | binäre Eingabe funktioniert für kleine, schnelle Bälle |
| Go2-Zeitbudget | REBot 2025, https://arxiv.org/abs/2508.06229 | < 0,5 s Vorwarnung: 5 %; 0,5–1,5 s: 65 % Erfolg | Trigger muss ≤ 100 ms nach dem Abwurf kommen |
| SNN-Detektion | Ziegler et al. 2024, https://arxiv.org/abs/2403.10677 | 0,6–0,9 ms auf neuromorphem Chip | dichte Positions-Supervision aus automatischen Labels |
| Dodgeball | Humanoider Dodgeball (Unitree G1) 2026, https://arxiv.org/abs/2611.01197 | > 85 % Ausweichen | prozedurale Ballspuren, harte Negative |
| Loss | MIL / Max-Pooling-Loss, Amazon 2022, https://arxiv.org/abs/2206.07261 | – | Max über die Zeit wie im Live-Trigger |
| Loss | Hey Snips 2019, https://arxiv.org/abs/1811.07684 | 0,12 % FRR bei 0,5 FA/h | Ziel kurz nach dem Ereignis, Rest maskiert |
| SNN-Streaming | SynSense Xylo KWS 2024, https://arxiv.org/abs/2406.15112 | 8-Bit-Gewichte, 16-Bit-Zustand | nächstes Analogon: binärer Streaming-Detektor mit wenig Daten |
| Readout | PLIF / Populationsausgang, https://arxiv.org/abs/2007.05785 | – | mehrere Ausgangsneuronen pro Klasse für feinere Evidenz |
| Frühe Entscheidung | SPRT-TANDEM, ICLR 2021, https://arxiv.org/abs/2006.05587 | – | CUSUM/SPRT statt Fenstersumme |
| Quantisierung | QSNN / MINT, https://arxiv.org/abs/2202.07221 | – | Fixpunkt-Risiken: Clipping bei ±2 bzw. ±8 |
| Transfer | LP-FT, ICLR 2022, https://arxiv.org/abs/2202.10054 | +10 % außerhalb der Verteilung | erst nur den Kopf trainieren, dann alles |
| Augmentation | ShapeAug 2024, https://arxiv.org/abs/2401.02274 | +2–6 % | prozedurale Ball-Generatoren |
| Augmentation | EVDodgeNet, ICRA 2020, https://arxiv.org/abs/1906.02919 | 70 % Ausweichen, nur Sim-trainiert | Transfer von synthetisch auf real |
| Augmentation | Copy-Paste (Dwibedi 2017 u. a.), https://arxiv.org/abs/1708.01642 | +7–9 % bei kleinen Objekten | Basis für `--synth-throw` |
| Kontrafaktisch | Teney et al., ECCV 2020, https://arxiv.org/abs/2004.09034 | – | Paare Wurf / Wurf ohne Ball |
| Sim-to-Real | Stoffregen et al., ECCV 2020, https://arxiv.org/abs/2003.09078 | +10–40 % | Kontrast und Rauschen breit variieren |
| Event-Augmentation | NDA, ECCV 2022, https://arxiv.org/abs/2203.06145 | +10–14 % | nur kleine Zusatzgewinne erwartet |
| Daten aus Video | v2e, https://arxiv.org/abs/2006.07722 | – | zusätzliche Würfe aus Zeitlupen-Videos |
| Bio-inspiriert | OMS-Retina-Segmentierung 2024, https://arxiv.org/abs/2408.09454 | mIoU 80 % | feste DoG-Schicht für kleine bewegte Objekte |
| Bio-inspiriert | Neuromorpher Pinball-Detektor 2026, https://arxiv.org/abs/2609.24403 | 21,7 ms Reaktion, ohne Training | analytisch gesetzter Ball-Detektor in conv+LIF |
| Bio-inspiriert | Spiking LGMD (Salt et al.), https://arxiv.org/abs/1910.07960 | feuert auch auf Handbewegung | reine Looming-Detektoren reichen nicht |
| Bio-inspiriert | TDE-3, https://arxiv.org/abs/2402.11662 | – | Richtungs-Veto (Barlow-Levick) |
| Sensor | Prophesee-Bias- und Filter-Doku, https://docs.prophesee.ai/stable/hw/manuals/biases.html | < 150 µs Sensor-Latenz | ERC/STC als mögliche Ursache für den unsichtbaren Ball |
