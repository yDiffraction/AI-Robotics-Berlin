# _dataset01_010203

Gelabelte, in Clips geschnittene Fassung der eigenen Aufnahmen `dataset/01`, `02` und `03` (GenX320 am Kria, 2026-11-21).
Eigene Aufnahmen, nicht der Durance-Datensatz.

| Klasse | Ordner | ID | Clips | Bedeutung |
|---|---|---|---|---|
| throw | `throw/` | 01 | 49 | richtiger Wurf, der Ball verlässt die Hand |
| feint | `feint/` | 02 | 68 | Fake-Wurf, dieselbe Bewegung ohne Loslassen |
| idle | `idle/` | 03 | 255 | keine Wurfbewegung (Stehen, Gehen, Ball aufheben, leere Szene) |

## Clips

- Dateiname: `gesture_<ID>-<label>-d<Datensatz>_<Uhrzeit der Aufnahme>-<tag>.bin`.
  `t01234` = Wurfbewegung mit Entscheidungszeitpunkt bei 12,34 s der Aufnahme, `g01234` = Idle-Clip aus einer Pause, Start bei 12,34 s.
- Jeder Clip ist 1,2 s lang, Zeitstempel ab 0. Bei throw und feint liegt der Entscheidungszeitpunkt (Maximum der Eventrate im Schwung) bei 0,7 s, also `release_us` = 700000.
- Format wie die Aufnahmen: 4 Byte Länge, dann Events `x, y, polarity, timestamp`. Lesbar mit `tools/events_tool.py` und `river-tools play`.
- `preview_<label>_<n>.png`: eine Zeile je Clip, 12 Bilder zu je 100 ms (ON grün, OFF magenta).

## labels.csv

Eine Zeile je Clip: `file, folder, label, class_id, dataset, source, parent, start_s, decision_s, release_us, events, verified, note`.

- `source` ist die Aufnahme, aus der der Clip stammt, `start_s` und `decision_s` sind Sekunden in dieser Aufnahme.
- `verified`: `1` = von Noah im Label-Tool geprüft, `claude` = von Claude anhand von Bildfolgen gelabelt, `auto` = Idle-Clip aus der Pause zwischen zwei Bewegungen.
- Clips mit demselben `parent` stammen aus derselben Bewegung oder derselben Pause.

## Herkunft

| Datensatz | Aufnahme | throw | feint | idle |
|---|---|---|---|---|
| 01 | 25 kurze Dateien, 10 Bewegungen behalten | 7 | 3 | 5 |
| 02 | 170441 | 5 | 10 | 32 |
| 03 | 195808 | 6 | 4 | 21 |
| 03 | 200001 | 6 | 19 | 48 |
| 03 | 200828 | 14 | 32 | 119 |
| 03 | 201452 | 5 | 0 | 24 |
| 03 | 201622 | 4 | 0 | 4 |
| 03 | 201703 | 2 | 0 | 2 |

## Zu beachten

- Train und Test nach `source` oder `dataset` trennen, nicht nach Clip: Clips aus derselben Aufnahme sehen sich sehr ähnlich.
- Zwei Drittel der Clips sind idle. Beim Training Klassen ausgleichen (`training/train_throw_own.py` zieht jede Klasse gleich oft).
- 24 Idle-Clips enthalten fast keine Events (leere Szene).
- Als `invalid` markierte Bewegungen (15 in 01, 120 in 03) fehlen, auch als Idle-Clip. Abschnitte, in denen die Kamera bewegt wurde, sind abgeschnitten.

## Neu erzeugen

Die Labels stehen in `dataset/<NN>/samples/labels.csv` und lassen sich mit `python tools/label_tool.py` ändern. Danach:

```bash
python tools/build_dataset.py
```
