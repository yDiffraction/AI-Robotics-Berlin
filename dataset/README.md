# Eigene Aufnahmen (GenX320 am Kria)

Aufgenommen am 2026-11-21 mit `river-tools capture --host root@durance-kria.local` (Start/Stop von Hand, 1,5–3 s je Datei).
Eigene Aufnahmen, nicht der Durance-Datensatz.

| Ordner | Dateien | Inhalt |
|---|---|---|
| `01/samples/` | 25 | je eine Wurfbewegung, 1,5–3 s |
| `02/samples/` | 1 | lange Aufnahme mit 15 Wurfbewegungen |
| `03/samples/` | 6 | lange Aufnahmen mit 92 brauchbaren Wurfbewegungen |
| `04/samples/` | 2 | mit `tools/record_tool.py`: Idle-Aufnahme ohne Wurfbewegung (5:47 min) und Wurf-Aufnahme mit 69 Hinweisen (5:18 min); die Aufnahmen selbst sind zu groß für GitHub, nur `hints.csv` und `labels.csv` sind eingecheckt |
| `_dataset01_01020304/` | 469 Clips | aktuelle Version: gelabelt und in 1,2-s-Clips geschnitten (throw, feint aus 01 bis 04, idle nur aus der Idle-Aufnahme in 04), siehe README dort |
| `_dataset01_010203/` | 372 Clips | Vorgänger ohne 04, idle aus den Pausen zwischen den Würfen |
| `unsorted/samples/` | 1 | erste Testaufnahme (8 s) |

Format wie beim Durance-Datensatz (4 Byte Länge, dann Events `x, y, polarity, timestamp`), also direkt lesbar mit:

```bash
uv run tools/events_tool.py info dataset/01/samples/*.bin
river-tools play dataset/01/samples/<datei>.bin
```

## Aufnehmen mit Hinweisen

```bash
python tools/record_tool.py                       # Aufnahmen nach dataset/04/samples, Seite http://127.0.0.1:8766
python tools/record_tool.py --out dataset/05/samples --host root@durance-kria.local
```

Start und Stop per Button, die ganze Aufnahme landet als eine `sample_<datum>_<zeit>.bin` im Ordner. Während der Aufnahme ungefähr im Moment des Wurfs eine Taste drücken: Leertaste für einen richtigen Wurf, `k` für einen Fake-Wurf, Backspace löscht den letzten Hinweis. Die Hinweise stehen in `hints.csv` im selben Ordner (Datei, `throw` oder `feint`, Sekunden ab dem ersten Event der Datei, Uhrzeit). Auf dem Board darf dabei keine andere Kamera-App laufen (`despot_send_and_forget` vorher beenden).

## Labeln

In jedem `samples/`-Ordner liegt eine `labels.csv` mit einer Zeile je Wurfbewegung: Datei, Label (`throw` richtiger Wurf, `feint` Fake-Wurf, `idle`, `invalid`, `end`), Entscheidungszeitpunkt in Sekunden und `verified`.

```bash
python tools/find_swings.py dataset/04/samples/*.bin   # neue Aufnahmen: Bewegungen suchen, Label "?"
python tools/label_tool.py                        # Clips im Loop ansehen, per Hotkey labeln (http://127.0.0.1:8765)
python tools/build_dataset.py --idle-from dataset/04/samples/sample_20261121_211821_487061.bin   # Clips nach dataset/_dataset01_01020304 schneiden, idle nur aus dieser Aufnahme
```

`training/propose_labels.py` trägt für offene Zeilen Vorschläge eines trainierten Modells ein. Sie zählen erst, wenn sie im Label-Tool bestätigt sind.
