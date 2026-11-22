# GOAL B – Früherkennung auf ganzen Aufnahmen (runs/early_ctx)

Stand 22.11.2026, 12:10. **Gate-Urteil: NICHT BESTANDEN.** Keine Variante erreicht auf zurückgehaltenen ganzen Aufnahmen ≥ 60 % der Wurfbewegungen bei ≤ 2–3 Pausen-FA/min mit Lead ≤ −200 ms. Das gilt auch dann, wenn man nur zählt, was über der Zufallsbaseline liegt.

## 1. Was ich geändert habe und warum

**Eval** (`ctx.py score`, ersetzt `tools/early_stream.py`):
- **Zustandsbehaftet, ohne Reset.** Die ganze Aufnahme läuft als *ein* Lauf, mit Reset nur am Dateianfang (`stateful_forward`). Die Gleichheit mit dem vollen Sequenz-Forward ist geprüft: 0 abweichende Frames von 1169.
  - Meine erste Eval nutzte noch den 30-s-Reset von `early_stream.py`. Diese Zahlen habe ich verworfen.
- **Pausenzeit als Vereinigung von Masken** und nicht als Summe. Die Aufnahme wird beim `end`-Label abgeschnitten.
- **Jeder Alarm landet in genau einer Kategorie:**
  - Treffer: erster unbenutzter Alarm im Fenster [Peak − 1,0 s, Peak + 0,3 s], 1:1 zugeordnet.
  - Hard-Neg-FA: innerhalb ±1 s um `idle`/`invalid`-Labels, also Gehen, Ball aufheben, Kamerahandling.
  - FA nahe Schwung: ±1,5 s um einen Schwung, einschließlich Doppelalarmen.
  - Pausen-FA: alles andere.
- **Nur frische Alarme zählen.** Ein Alarm ist ein Aufwärtsdurchgang durch θ mit 1 s Refraktärzeit. Was schon über θ liegt, ist kein Treffer. Die Spalte „y ≥ θ bei Fensterbeginn“ zeigt, wie oft das vorkam.
- **Zufallsbaseline:** dieselben Alarme, alle Label-Zeiten pro Aufnahme zirkulär um einen zufälligen Offset verschoben, 100 Ziehungen. Angegeben sind Mittel und 95. Perzentil der Treffer.
- **Lead:** gegen `release_ms`, wo vorhanden (33/42 Würfe in 02/03), sonst gegen den Peak, in getrennten Spalten. Negativ heißt früh.
- **θ wird pro Fold nur auf den Trainings-Sessions dieses Folds gewählt** (02/03, ohne 215305). Regel, vor allen Ergebnissen festgelegt: maximaler Recall unter Pausen-FA ≤ 1,0/min. Wird das nirgends erreicht, gilt das kleinste FA.
  - Als Sensitivität, klar *post hoc*: dieselbe Regel mit ≤ 2,5/min.
- **Ein Review hat 3 Fehler in meiner ersten Eval gefunden, alle behoben:**
  - Alarme nahe Schwüngen wurden nirgends gezählt.
  - Hard-Negatives neben Schwüngen wurden ausgeblendet.
  - Ein Alarm konnte zwei Schwünge treffen.
- **Folds unverändert** (f0 = 170441, 201703, 215305; f1 = 195808, 200001, 200828, 201452, 201622). Nur so bleiben die Baselines held-out. 211821 dient nur als Trainings-Negativ und wird nie bewertet.

**Daten und Training** (`ctx.py train`, Tims v1-Architektur, 1 Ausgang, DVSG-Init, τ ≤ 8, EMA-Loss wie bei Tim):
- Trainiert wird auf **Fenstern (3 s + 0,6 s Einschwingen) aus den ganzen Aufnahmen** der Trainings-Sessions statt auf geklebten Clips.
  - Damit sind echte Pausen, Gehen, Aufheben und Kamerahandling der Wurfszene Negative.
  - `idle`/`invalid` haben Gewicht 2 (bei `whneg` 4).
  - Ziel ist 1 auf [Release − 0,5 s, Release + 0,3 s], wo kein Release bekannt ist auf den Peak.
  - Dazu kommt 211821 als Ruhe-Negativ. Augmentiert wird leicht: Spiegeln, x-Shift, Ausdünnen, Polaritätstausch, Rauschen.
  - Das testet Ursache 1 (fehlende Pausen-Negative) und Ursache 2 (Lernen auf geklebten Strömen).
- Die Frames stammen aus `prep.py`, das bitgleich mit `ball_live.frames_from_file` bint.
- **Ursache 3 (Vorverarbeitung) ist geprüft und widerlegt:** Tims Bank-Clips und das Live-Binning decken sich zu 96–99,9 % (IoU, Versatz 0, 12 Clips).
- **Varianten**, je eine Hypothese:
  - `whole`: nur ganze Aufnahmen. 2 Seeds (s0, s1).
  - `whfor`: wie `whole`, aber 25 % jedes Batches sind fremde Negative aus Tims Banks (mb/dd/dvsg). Seed s0, s1 siehe unten.
  - `whneg`: `whole_s0` weitertrainiert, 600 Schritte, stärkere Negativgewichtung (p_pos 0,3, hn_w 4, pos_w 1). Hypothese: Die Pausen sind unterangepasst, denn selbst in-sample lag Pausen-FA bei θ 0,8 noch bei 5,7/min.
  - `_c1`: **nachträglicher Bias-Clamp gegen tonische Kanäle.** Der BN-gefaltete Bias wird auf ≤ 0,9·v_th/τ begrenzt, ohne Nachtraining. Danach ist kein Kanal mehr tonisch (Zählung `tonic_report`).
- **Readouts:**
  - `abs`: EMA mit α 0,85 (Tim).
  - `slow`: EMA mit α 0,93.
  - `rel`: y minus eigener Mittelwert über [t − 1,7 s, t − 0,2 s], also der Anstieg.

## 2. Ergebnisse (zustandsbehaftet, 02/03 held-out, 107 Schwünge = 42 Würfe + 65 Täuschungen)

Gemeinsamkeiten aller Zeilen:
- Echte Pausenzeit 3,02 min, Hard-Neg-Zeit 2,87 min, Gesamtdauer 10,2 min.
- Trefferfenster [−1,0, +0,3] s um den Peak.
- FA-Obergrenze: einseitig 95 %, Poisson.
- 215305 steht separat, mit Fold-0-Modell, Tastendruck-Labels und unsichtbarem Ball.

### 2a. Hauptprotokoll (θ-Regel FA ≤ 1/min auf Training, vorab festgelegt)

| Modell | Readout | θ f0/f1 | Würfe | Täusch. | gesamt | Zufall (Mittel, p95) | Pausen-FA (/min, ≤95 %) | Hard-Neg-FA /min | FA nahe Schwung | Lead vs Release (median, n) | Lead vs Peak | y≥θ bei Fensterbeginn | 215305 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r2_old_fa (Baseline) | abs | .99/.99 | 22/42 | 38/65 | **60/107 (56 %)** | 40 (46) | 28 → **9,27** (≤12,7) | 18,8 | 32 | −80 (13) | −475 | 17 | 5/65, 1,8 FA/min |
| r2_ext_fa (Baseline) | abs | .99/.99 | 17/42 | 28/65 | **45/107 (42 %)** | 22 (27) | 9 → **2,98** (≤5,2) | 10,8 | 8 | **+145** (10) | +20 | 5 | 4/65, 1,1 |
| r2_old_fa + Clamp | abs | .99/.97 | 12/42 | 29/65 | 41/107 (38 %) | 29 (35) | 14 → 4,64 | 15,3 | 31 | −365 (8) | −520 | 9 | 1/65 |
| r2_ext_fa + Clamp | abs | .99/.99 | 18/42 | 22/65 | 40/107 (37 %) | 22 (28) | 6 → 1,99 | 11,2 | 14 | +190 (11) | +140 | 3 | 5/65 |
| whole_s0 | abs | .99/.97 | 3/42 | 6/65 | 9/107 (8 %) | 6 (9) | 0 → 0,00 (≤1,0) | 4,2 | 3 | −515 (2) | −570 | 0 | 0/65, 0 FA |
| whole_s1 | abs | .99/.97 | 7/42 | 11/65 | 18/107 (17 %) | 9 (13) | 1 → 0,33 | 4,5 | 4 | −470 (2) | −320 | 0 | 0/65 |
| whole_s1 | rel | .90/.70 | 8/42 | 17/65 | 25/107 (23 %) | 7 (11) | 3 → 0,99 (≤2,6) | 1,7 | 0 | −410 (7) | −430 | 0 | 0/65 |
| whfor_s0 | abs | .99/.90 | 11/42 | 16/65 | 27/107 (25 %) | 12 (17) | 3 → 0,99 (≤2,6) | 5,6 | 3 | −290 (7) | −160 | 0 | 0/65 |
| whfor_s0 | slow | .99/.75 | 13/42 | 15/65 | 28/107 (26 %) | 11 (15) | 3 → 0,99 | 4,5 | 4 | +60 (7) | −285 | 0 | 0/65 |
| whole_s0 + Clamp | abs | .99/.80 | 12/42 | 12/65 | 24/107 (22 %) | 11 (15) | 1 → 0,33 | 5,6 | 6 | −370 (7) | +20 | 1 | 0/65 |
| whneg_s0 | abs | .99/.95 | 5/42 | 5/65 | 10/107 (9 %) | 5 (7) | 0 → 0,00 (≤1,0) | 1,7 | 2 | +45 (4) | −440 | 0 | 0/65 |
| whneg_s0 | slow | .97/.65 | 19/42 | 16/65 | 35/107 (33 %) | 16 (21) | 3 → 0,99 (≤2,6) | 8,7 | 7 | +65 (14) | −30 | 2 | 0/65 |
| whneg_s0 + Clamp | rel | .90/.55 | 17/42 | 17/65 | 34/107 (32 %) | 12 (16) | 5 → 1,66 (≤3,5) | 4,2 | 5 | −435 (12) | −410 | 1 | 0/65 |
| whfor_s1 | abs | .99/.98 | 2/42 | 2/65 | 4/107 (4 %) | 3 (5) | 0 → 0,00 (≤1,0) | 1,4 | 0 | −450 (2) | −90 | 0 | 0/65 |
| whfor_s1 | rel | .90/.70 | 2/42 | 9/65 | 11/107 (10 %) | 5 (8) | 1 → 0,33 | 2,4 | 3 | +200 (1) | −330 | 0 | 0/65 |

Der `rel`-Readout der Baselines ergibt 1/107 (θ-Wahl auf Training kollabiert).

### 2b. Sensitivität, post hoc: θ-Regel Pausen-FA ≤ 2,5/min auf Training

| Modell | Readout | θ f0/f1 | gesamt | Zufall (Mittel, p95) | Pausen-FA /min (≤95 %) | Hard-Neg-FA /min | Lead vs Release |
|---|---|---|---|---|---|---|---|
| r2_ext_fa | abs | .99/.97 | 57/107 (53 %) | 30 (35) | 3,64 (≤6,0) | 16,0 | +80 |
| whole_s1 | rel | .80/.65 | 33/107 (31 %) | 11 (15) | 2,32 (≤4,4) | 3,1 | −415 |
| whole_s1 | abs | .99/.90 | 40/107 (37 %) | 17 (23) | 0,99 (≤2,6) | 7,7 | +80 |
| **whfor_s0** | **rel** | .80/.60 | **40/107 (37 %)** | 14 (19) | **1,66** (≤3,5) | 4,2 | **−505** (n=16) |
| whole_s0 | abs | .99/.85 | 33/107 (31 %) | 16 (21) | 0,99 | 9,8 | +75 |
| whfor_s1 (Seed 1) | rel | .85/.70 | 12/107 (11 %) | 6 (9) | 0,33 | 2,4 | −240 (n=2) |
| whfor_s1 (Seed 1) | abs | .97/.90 | 19/107 (18 %) | 9 (12) | 0,00 | 3,8 | −135 |
| whneg_s0 | slow | .80/.65 | 35/107 (33 %) | 16 (21) | 0,99 | 8,7 | +50 |

Zur Einordnung, nur berichtet und nicht zur Auswahl benutzt: Der beste Punkt mit θ auf dem *Test* gewählt (Orakel) ist whfor_s0 abs θ 0,70 mit 68/107 (64 %) bei 2,65 Pausen-FA/min und Lead −445 ms. Er hat aber 16,4 Hard-Neg-FA/min und 34 FA nahe Schwüngen, und die Zufallsbaseline liegt bei 34 (p95 42). Das sind +34 Treffer über Zufall, mehr als jede Baseline (+20 bis +23). **Seed 1 bestätigt das:** whfor_s1 abs θ 0,70 kommt auf 66/107 bei Zufall 31 (p95 37), 3,3 Pausen-FA/min, 12,6 Hard-Neg-FA/min und Lead −290 ms. Der Abstand zeigt: Die θ-Wahl auf den unruhigen Trainings-Sessions (03, viel Kamerahandling) ist hier ein Engpass. `oracle_chance.py` reproduziert die Zahl.

## 3. Gate-Urteil: nicht bestanden

Gemessen an Treffern über dem Zufallsmittel liegen alle Varianten bei etwa +20 bis +26 von 107. Das sind 19–24 %, weit unter 60 %.

**Baselines:**
- Zustandsbehaftet erreicht r2_old_fa zwar 56 %, aber bei 9,3 Pausen-FA/min. Davon sind 40 Treffer Zufall.
- r2_ext_fa löst im Median nach dem Release aus (+145 ms).

**Neue Varianten:**
- **Training auf ganzen Aufnahmen** senkt Pausen- und Hard-Neg-FA deutlich: 0–1,7 statt 3–9 Pausen-FA/min, 2–6 statt 11–19 Hard-Neg-FA/min.
- Die Leads sind konsistent früh: −400 bis −500 ms mit `rel`.
- **Der Recall bricht aber ein.** Mit der vorab festgelegten θ-Regel erreichen sie 8–26 % gegenüber 42–56 % der Baselines.
- Bester ehrlicher Punkt (2b, post hoc): whfor_s0 rel mit 37 %, etwa 2,7× Zufall, 1,7 Pausen-FA/min und −505 ms Lead. Das besteht das Gate nicht und **repliziert mit Seed 1 nicht** (12/107). Die θ-Wahl ist über Seeds instabil.

**Weitere Beobachtungen:**
- Der **Bias-Clamp** (tonische Kanäle → 0) senkt FA und Zufallsrate der Baselines, hebt aber den Recall über Zufall nicht.
  - Der Drift-Test (60 s Rauschen) liefert nach dem Clamp bei r2_old_fa f1 noch 12 statt 55 Ausgangs-Spikes, bei whole_s0 f1 aber 56 statt 16. Der Clamp allein reicht also nicht.
  - Auch die neuen Modelle haben tonische Kanäle (L2 bis 22/64), weil Tims Training sie nicht bestraft.
- **215305** (Fold-0-Modelle, trainiert auf 03): Die neuen Modelle sind dort stumm, 0/65 bei 0 FA. Die Baselines kommen auf 4–5/65. Das passt zu „anderes Setting, Ball unsichtbar, Labels per Tastendruck“.
- **Fold 0 hat kaum Test-Schwünge** (17 in 02/03). Die θ-Wahl für Fold 1 stützt sich auf nur 0,4 min Pause (170441 + 201703).

**Diagnose:**
- Die echten Negative der Wurfszene sind der richtige Hebel gegen „Person bewegt sich“.
- Ein Netz mit τ ≤ 80 ms und 800 Schritten trennt Ausholen und Gehen oder Aufheben aber noch nicht. Selbst in-sample bleibt die Pausen-FA hoch.
- Einziges positives Signal, nur mit Orakel-θ und daher keine Kennzahl: whfor (ganze Aufnahmen plus fremde Negative) liegt bei festem θ 0,70 in beiden Seeds bei +34/+35 Treffern über Zufall (66–68/107, Lead −290 bis −445 ms). Die Baselines kommen auf +20 bis +23. Der Preis sind 2,7–3,3 Pausen-FA/min und 12–16 Hard-Neg-FA/min. Für einen ehrlichen Test bräuchte es eine eigene Validierungs-Session zur θ-Wahl.

**Nächste Schritte, nicht heute:**
- Mehr Schritte.
- Bias-Strafterm im Training statt nachträglichem Clamp.
- Mehr Pausen-Clips aus 02/03, also `_dataset01_010203`.
- Längere Zeitkonstanten, sofern exportierbar.

## 4. Checkpoints, θ, Readout

- **Lokal:** `runs/early_ctx/runs/<variante>/model_f{0,1}.pt`, also whole_s0, whole_s1, whfor_s0, whfor_s1 und whneg_s0. Dazu die Trainingslogs `trainlog_f*.json`.
- **Ausgaben:**
  - zustandsbehaftete Streams: `runs/early_ctx/sstreams/<variante>_c{0,1}/f{fold}_{rec}.npz`, mit `info.json` (Tonik, Drift, Sanity)
  - Scores: `score_stateful.json` (2a) und `score_stateful_fa25.json` (2b)
- **A100:** `~/early_ctx/runs/<variante>/model_f{0,1}.pt`, Streams unter `~/early_ctx/sstreams/`, Logs unter `~/early_ctx/logs/`. Tims Dateien sind unverändert. Unter `~/early_ctx/tim/` liegen nur Kopien.
- **Falls überhaupt etwas weiterverfolgt wird:** whfor_s0 mit Readout `rel` (Anstieg über dem Mittel von [t − 1,7, t − 0,2] s der EMA mit α 0,85), θ 0,80 (Fold-0-Modell) bzw. 0,60 (Fold-1-Modell), Refraktärzeit 1 s.
  - Das ist **kein freigegebenes Demo-Modell.** Exportiert wurde nichts.

## 5. Reproduzieren

```bash
# lokal (Haupt-Checkout): Frames + labels.json
python3 runs/early_ctx/prep.py
rsync -az runs/early_ctx/{frames,labels.json,ctx.py,table.py} team-1-user@<A100>:early_ctx/
# A100 (einmalig: Kopien von Tims Code, nichts verändern)
mkdir -p ~/early_ctx/tim && cp ~/early/early_cv.py ~/robust/snn_model.py ~/robust/speedpatch.py ~/early_ctx/tim/
cd ~/early_ctx; PY=~/snn_pretrain/.venv/bin/python
for f in 0 1; do
  $PY -W ignore ctx.py train --fold $f --out runs/whole_s0 --seed 0
  $PY -W ignore ctx.py train --fold $f --out runs/whole_s1 --seed 1
  $PY -W ignore ctx.py train --fold $f --out runs/whfor_s0 --seed 0 --foreign 0.25
  $PY -W ignore ctx.py train --fold $f --out runs/whfor_s1 --seed 1 --foreign 0.25
  $PY -W ignore ctx.py train --fold $f --out runs/whneg_s0 --seed 0 --resume runs/whole_s0 --steps 600 --lr 5e-4 \
      --p-pos 0.3 --hn-w 4 --pos-w 1.0 --early-w 1.5
done
for r in whole_s0 whole_s1 whfor_s0 whfor_s1 whneg_s0; do for c in 0 1; do
  $PY -W ignore ctx.py stream --model-dir runs/$r --out sstreams/${r}_c$c --clamp $c; done; done
for r in r2_old_fa r2_ext_fa; do for c in 0 1; do
  $PY -W ignore ctx.py stream --model-dir ~/early/runs_ext/$r --out sstreams/${r}_c$c --clamp $c; done; done
$PY -W ignore ctx.py score --n-chance 100 --streams sstreams/* --json score_stateful.json            # 2a
$PY -W ignore ctx.py score --n-chance 100 --fa-target 2.5 --streams sstreams/* --json score_stateful_fa25.json  # 2b
$PY table.py score_stateful.json
```

## 6. Grenzen

- **Seeds:** whole hat 2 Seeds, und die streuen stark (abs 9 vs. 18 Treffer, `rel` 5 vs. 25). whfor hat 2 Seeds: s0 kommt mit gewähltem θ auf 25–37 %, s1 auf 4–18 %. Bei festem θ 0,70 sind beide ähnlich (68 bzw. 66/107, Zufall 34 bzw. 31). whneg hat nur 1 Seed.
  - Unterschiede unter etwa 10 Treffern sind nicht belastbar.
- **Pausenzeit:** nur 3,0 min echte Pause in 02/03. Die Poisson-Obergrenze bei 0 FA liegt deshalb schon bei 1,0/min, und die FA-Raten sind grob.
  - Die θ-Wahl von Fold 1 beruht auf 0,4 min Pause.
- **In-sample:** θ wird auf den Trainings-Sessions gewählt, die die Modelle gesehen haben. Das ist optimistisch für FA, und das Modell reagiert dort zu stark, weshalb θ meist 0,99 wird. Eine eigene Validierungs-Session gab es nicht, dafür sind es zu wenige Sessions.
  - 2b ist post hoc (die FA-Schwelle wurde nach Sicht auf die Testergebnisse erhöht). Die Orakel-Zahl ist ausdrücklich keine Kennzahl.
- **Unvollständig:** `sstreams/whfor_s0_c1` und `sstreams/whole_s1_c1` wurden aus Zeitgründen abgebrochen (je 3 Dateien) und sind nicht bewertet. Für `score` deshalb die Verzeichnisse einzeln angeben statt `sstreams/*`. `obsolete_segmented/` enthält die verworfene erste Runde mit 30-s-Reset.
- **Labels:** 215305 hat Tastendruck-Labels. Die `idle`/`invalid`-Fenster sind ±1 s um Punktlabels und keine echten Segmente. 9 der 42 Würfe haben kein `release_ms` und gehen nur in den Peak-Lead ein.
- **Zufallsbaseline:** nur für die gewählten θ berechnet. Die zirkuläre Verschiebung erhält die Alarmrate, aber nicht die Kopplung „Aktivität ↔ Schwung“. Sie ist also eine untere Schranke dafür, was ein reiner Bewegungsdetektor trifft.
