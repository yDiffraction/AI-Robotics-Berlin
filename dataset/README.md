# Eigene Aufnahmen (GenX320 am Kria)

Aufgenommen am 2026-11-21 mit `river-tools capture --host root@durance-kria.local` (Start/Stop von Hand, 1,5–3 s je Datei).
Eigene Aufnahmen, nicht der Durance-Datensatz.

| Ordner | Dateien | Inhalt |
|---|---|---|
| `01/samples/` | 25 | noch ohne Klassen-Label im Dateinamen |
| `unsorted/samples/` | 1 | erste Testaufnahme (8 s) |

Format wie beim Durance-Datensatz (4 Byte Länge, dann Events `x, y, polarity, timestamp`), also direkt lesbar mit:

```bash
uv run tools/events_tool.py info dataset/01/samples/*.bin
river-tools play dataset/01/samples/<datei>.bin
```

Noch offen: Dateien nach dem Schema aus `recording/PROTOCOL.md` benennen und in `recording/takes.csv` eintragen.
