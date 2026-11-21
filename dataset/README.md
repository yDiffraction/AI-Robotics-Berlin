# Eigene Aufnahmen (GenX320 am Kria)

Aufgenommen am 2026-11-21 mit `river-tools capture --host root@durance-kria.local` (Start/Stop von Hand, 1,5–3 s je Datei).
Eigene Aufnahmen, nicht der Durance-Datensatz.

| Ordner | Dateien | Inhalt |
|---|---|---|
| `01/samples/` | 25 | je eine Wurfbewegung, 1,5–3 s |
| `02/samples/` | 1 | lange Aufnahme mit 15 Wurfbewegungen |
| `03/samples/` | 6 | lange Aufnahmen mit 92 brauchbaren Wurfbewegungen |
| `_dataset01_010203/` | 372 Clips | gelabelt und in 1,2-s-Clips geschnitten (throw, feint, idle), siehe README dort |
| `unsorted/samples/` | 1 | erste Testaufnahme (8 s) |

Format wie beim Durance-Datensatz (4 Byte Länge, dann Events `x, y, polarity, timestamp`), also direkt lesbar mit:

```bash
uv run tools/events_tool.py info dataset/01/samples/*.bin
river-tools play dataset/01/samples/<datei>.bin
```

## Labeln

In jedem `samples/`-Ordner liegt eine `labels.csv` mit einer Zeile je Wurfbewegung: Datei, Label (`throw` richtiger Wurf, `feint` Fake-Wurf, `idle`, `invalid`, `end`), Entscheidungszeitpunkt in Sekunden und `verified`.

```bash
python tools/find_swings.py dataset/04/samples/*.bin   # neue Aufnahmen: Bewegungen suchen, Label "?"
python tools/label_tool.py                        # Clips im Loop ansehen, per Hotkey labeln (http://127.0.0.1:8765)
python tools/build_dataset.py                     # Clips nach dataset/_dataset01_010203 schneiden
```

`training/propose_labels.py` trägt für offene Zeilen Vorschläge eines trainierten Modells ein. Sie zählen erst, wenn sie im Label-Tool bestätigt sind.
