# Paper (LaTeX, plantilla IEEE Conference)

Versión LaTeX del paper del proyecto, con la plantilla **IEEE Conference** (`IEEEtran`, modo
`conference`). Es el mismo contenido que la [página interactiva](https://fluchetti45.github.io/deepracer/)
y el README raíz, en formato de dos columnas listo para PDF.

- **`main.tex`** — el documento.
- **`references.bib`** — las 8 referencias.
- **`figures/`** — figuras (copiadas de `analysis/`): eval, saliencia y curvas de entrenamiento.
- **`main.pdf`** — el PDF compilado (6 páginas).

## Compilar

**Local** (requiere una distribución TeX con `IEEEtran`, p.ej. MiKTeX o TeX Live):

```bash
cd paper
latexmk -pdf main.tex        # corre pdflatex + bibtex + pdflatex las veces necesarias
```

**Overleaf:** subí la carpeta `paper/` (o creá un proyecto en blanco y arrastrá `main.tex`,
`references.bib` y `figures/`). Overleaf trae `IEEEtran` y compila sin configuración extra.
La plantilla base es la [IEEE Conference Template](https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn).

## Estructura del documento

Abstract · Introducción · Trabajo relacionado · Método (con diagrama TikZ del pipeline) ·
Setup experimental (hiperparámetros) · Resultados (tablas media±desvío, por-semilla y
significancia Mann-Whitney) · Análisis de saliencia · Limitaciones · Conclusión · Referencias.
