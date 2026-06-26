# Resultados agregados (eval held-out, media ± desvío sobre seeds)

_15 corridas · tracks held-out: track4.png, track5.png · 10 episodios/track (modo tasa de éxito)._

## Desempeño global (promedio de los tracks held-out)

| Variante | Lap rate (%) | Reward/ep | Off-track (%) | Tiempo vuelta (s) |
|---|---|---|---|---|
| Geometrica | 60.0 ± 23.5 | 565.4 ± 107.5 | 28.0 ± 17.9 | 166.8 ± 18.6 |
| Vision (1 frame) | 4.0 ± 8.9 | 195.8 ± 67.0 | 96.0 ± 8.9 | 238.0 ± 0.0 |
| Vision apilada (4) | 0.0 ± 0.0 | 42.8 ± 19.1 | 100.0 ± 0.0 | — |

## Lap rate por pista (%)

| Variante | track4.png | track5.png |
|---|---|---|
| Geometrica | 94.0 ± 13.4 | 26.0 ± 42.2 |
| Vision (1 frame) | 0.0 ± 0.0 | 8.0 ± 17.9 |
| Vision apilada (4) | 0.0 ± 0.0 | 0.0 ± 0.0 |

## Significancia — Mann-Whitney exacto (lap rate por seed)

| Comparación | U | p (two-sided) |
|---|---|---|
| Geometrica vs Vision (1 frame) | 25.0 | 0.0079 |
| Geometrica vs Vision apilada (4) | 25.0 | 0.0079 |
| Vision (1 frame) vs Vision apilada (4) | 15.0 | 1.0000 |
