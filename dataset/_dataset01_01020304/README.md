# _dataset01_01020304

Gelabelte, in Clips geschnittene Fassung der eigenen Aufnahmen `dataset/01` bis `04` (GenX320 am Kria, 2026-11-21). Throw und feint kommen aus 01 bis 03 und der Wurf-Aufnahme `04/215305`, idle nur aus der Idle-Aufnahme `04/211821`.
Eigene Aufnahmen, nicht der Durance-Datensatz. Vorgänger: `_dataset01_010203` (ohne 04, idle aus den Pausen zwischen den Würfen).

| Klasse | Ordner | ID | Clips | Bedeutung |
|---|---|---|---|---|
| throw | `throw/` | 01 | 77 | richtiger Wurf, der Ball verlässt die Hand |
| feint | `feint/` | 02 | 104 | Fake-Wurf, dieselbe Bewegung ohne Loslassen |
| idle | `idle/` | 03 | 288 | keine Wurfbewegung: die 5:47 min lange Idle-Aufnahme `04/211821`, lückenlos in Clips geschnitten |

## Clips

- Dateiname: `gesture_<ID>-<label>-d<Datensatz>_<Uhrzeit der Aufnahme>-<tag>.bin`.
  `t01234` = Wurfbewegung mit Entscheidungszeitpunkt bei 12,34 s der Aufnahme, `g01234` = Idle-Clip, Start bei 12,34 s der Aufnahme.
- Jeder Clip ist 1,2 s lang, Zeitstempel ab 0. Bei throw und feint liegt der Entscheidungszeitpunkt (Maximum der Eventrate im Schwung) bei 0,7 s, also `release_us` = 700000.
- Idle-Clips sind ausgedünnt: je 10 ms, Polarität und 4×4-Pixel-Block bleibt nur das erste Event. Das ist das Eingaberaster des Modells (`training/prep_clips.py`: 120 Bilder zu 10 ms, 80×80), das Modell sieht also dieselben Bilder wie beim vollen Clip. Die Idle-Clips brauchen so 138 MB statt 717 MB. Throw- und Feint-Clips enthalten alle Events.
- Format wie die Aufnahmen: 4 Byte Länge, dann Events `x, y, polarity, timestamp`. Lesbar mit `tools/events_tool.py` und `river-tools play`.
- `preview_<label>_<n>.png`: eine Zeile je Clip, 12 Bilder zu je 100 ms (ON grün, OFF magenta).

## labels.csv

Eine Zeile je Clip: `file, folder, label, class_id, dataset, source, parent, start_s, decision_s, release_us, events, verified, note`.

- `source` ist die Aufnahme, aus der der Clip stammt, `start_s` und `decision_s` sind Sekunden in dieser Aufnahme.
- `verified`: `1` = von Noah im Label-Tool geprüft, `claude` = von Claude anhand von Bildfolgen gelabelt, `recording` = Idle-Clip aus einer Aufnahme, die als Ganzes idle ist.
- Clips mit demselben `parent` stammen aus derselben Bewegung.

## Herkunft

| Datensatz | Aufnahme | throw | feint | idle |
|---|---|---|---|---|
| 01 | 25 kurze Dateien, 10 Bewegungen behalten | 7 | 3 | 0 |
| 02 | 170441 | 5 | 10 | 0 |
| 03 | 195808 | 6 | 4 | 0 |
| 03 | 200001 | 6 | 19 | 0 |
| 03 | 200828 | 14 | 32 | 0 |
| 03 | 201452 | 5 | 0 | 0 |
| 03 | 201622 | 4 | 0 | 0 |
| 03 | 201703 | 2 | 0 | 0 |
| 04 | 211821 (Idle-Aufnahme) | 0 | 0 | 288 |
| 04 | 215305 (mit `tools/record_tool.py`) | 28 | 36 | 0 |

## Aufnahme 04/215305

- Aufgenommen mit `tools/record_tool.py`: beim Wurf Leertaste, beim Fake-Wurf `k` gedrückt (`dataset/04/samples/hints.csv`, 69 Hinweise).
- Fünf Claude-Agenten haben jedem Hinweis den passenden Schwung zugeordnet und den Entscheidungszeitpunkt auf das Maximum der Eventrate gelegt (`verified` = `claude`, die Notiz nennt die Zeit des Hinweises). Das Label kommt aus der Taste, den Ball sieht man nur bei wenigen Würfen.
- Ungültig gesetzt: 1 Hinweis ohne Bewegung in der Nähe, 3 doppelte Hinweise auf denselben Schwung. Der Wurf bei 0,65 s fehlt, weil er zu nah am Dateianfang liegt.
- Unsicher: der Wurf bei 198,12 s (sieht eher nach Fake-Wurf aus), der Fake-Wurf bei 137,30 s (kaum Bewegung) und die Zuordnung bei 103,01, 105,13, 186,28 und 289,72 s.

## Zu beachten

- Train und Test nach `source` oder `dataset` trennen, nicht nach Clip: Clips aus derselben Aufnahme sehen sich sehr ähnlich.
- Gut 60 % der Clips sind idle. Beim Training Klassen ausgleichen (`training/train_throw_own.py` zieht jede Klasse gleich oft).
- Alle Idle-Clips stammen aus einer einzigen Aufnahme. Ein Holdout nach Aufnahme enthält deshalb entweder alle Idle-Clips oder keinen; ein Holdout nach Datensatz `04` enthält alle Idle-Clips.
- 127 Idle-Clips enthalten fast keine Events (unter 1000 nach dem Ausdünnen, leere Szene). In rund 90 Idle-Clips bewegt sich das ganze Bild, vor allem in den ersten 130 s und den letzten 40 s der Aufnahme. Der größte Idle-Clip hat 4,5 MB.
- Die als `idle` gelabelten Bewegungen und die Pausen zwischen den Würfen aus 01 bis 03 sind nicht enthalten. Die Labels stehen weiter in den `labels.csv` der Aufnahmen.
- Als `invalid` markierte Bewegungen (15 in 01, 120 in 03, 4 in 04) fehlen.

## Neu erzeugen

Die Labels stehen in `dataset/<NN>/samples/labels.csv` und lassen sich mit `python tools/label_tool.py` ändern. Danach:

```bash
python tools/build_dataset.py --idle-from dataset/04/samples/sample_20261121_211821_487061.bin
```

Ohne `--idle-from` kommen die Idle-Clips wieder aus den gelabelten Aufnahmen (als `idle` gelabelte Bewegungen und Pausen zwischen den Würfen).
