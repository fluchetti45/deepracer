"""
Progreso a lo largo de la pista (reward del supervisor, lado oraculo).

Los gates ordenados de un track definen una polilinea CERRADA (loop). Cada punto
del loop tiene una coordenada de arc-length `s in [0, L)`, con el orden de los
gates = sentido de la vuelta. El supervisor proyecta la posicion real del robot
sobre esa polilinea para obtener su `s`, y premia el avance `Δs` por step:
  - adelante (s creciente) -> +Δs ; atras -> -Δs ; loop en el lugar -> neto ~0.
Una vuelta completa = progreso neto acumulado >= L (volver al punto de spawn).

Todo en numpy puro, coords del MUNDO (x, y). No depende de la imagen/camara.
"""

import numpy as np


def build_loop(gates):
    """
    gates: lista ordenada de [x, y] (loop cerrado; NO repetir el primero al final).
    Devuelve (pts, cumlen, total): pts es (N+1, 2) con el primer punto repetido al
    final (cierre), cumlen[i] = arc-length acumulada hasta pts[i], total = perimetro.
    Devuelve None si hay menos de 2 gates o el perimetro es ~0.
    """
    if not gates or len(gates) < 2:
        return None
    pts = np.asarray(list(gates) + [gates[0]], dtype=float)  # cierra el loop
    seg = np.diff(pts, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    cumlen = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(cumlen[-1])
    if total <= 1e-9:
        return None
    return pts, cumlen, total


def project_s(loop, point):
    """
    Proyecta `point` (x, y) sobre la polilinea cerrada y devuelve el arc-length `s`
    del punto mas cercano. `loop` es la tupla de build_loop.
    """
    pts, cumlen, _ = loop
    p = np.asarray(point, dtype=float)
    best_d2 = np.inf
    best_s = 0.0
    for i in range(len(pts) - 1):
        a = pts[i]
        ab = pts[i + 1] - a
        seg_len2 = float(ab[0] * ab[0] + ab[1] * ab[1])
        if seg_len2 <= 1e-12:
            t = 0.0
        else:
            t = float(np.dot(p - a, ab) / seg_len2)
            t = max(0.0, min(1.0, t))
        proj = a + t * ab
        diff = p - proj
        d2 = float(diff[0] * diff[0] + diff[1] * diff[1])
        if d2 < best_d2:
            best_d2 = d2
            best_s = float(cumlen[i] + t * np.sqrt(seg_len2))
    return best_s


def signed_delta(s_prev, s_cur, total):
    """
    Delta de arc-length mas corto sobre el loop (maneja el wrap del cierre): el
    movimiento por step es chico, asi cruzar la 'meta' suma y no resta de golpe.
    """
    return (s_cur - s_prev + total / 2.0) % total - total / 2.0
