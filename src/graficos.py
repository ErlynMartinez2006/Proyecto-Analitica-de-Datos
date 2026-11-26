"""
graficos.py
===========
Estilo y funciones de gráficas compartidas por los cuadernillos del proyecto.
"""

import matplotlib.pyplot as plt

AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#8a8985"


def configurar_estilo() -> None:
    """Aplica el estilo visual del proyecto a todas las gráficas."""
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#c9c8c3", "axes.grid": True, "grid.color": "#e8e7e2",
        "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelcolor": "#52514e",
        "xtick.color": "#52514e", "ytick.color": "#52514e", "figure.dpi": 100,
    })


def barras(ax, etiquetas, valores, titulo, colores=AZUL, horizontal=False,
           ylabel=None, rotar=0, anotar=False) -> None:
    """Gráfica de barras verticales u horizontales; ``anotar`` escribe cada valor."""
    etiquetas = [str(e) for e in etiquetas]
    if horizontal:
        ax.barh(etiquetas, valores, color=colores, height=0.6)
        ax.grid(axis="y", visible=False)
    else:
        ax.bar(etiquetas, valores, color=colores, width=0.7)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", rotation=rotar)
    ax.set_title(titulo)
    if ylabel:
        ax.set_ylabel(ylabel)
    if anotar:
        for i, v in enumerate(valores):
            desplazamiento = 0.01 if v > 0 else -0.01
            ax.text(v + desplazamiento, i, f"{v:.2f}", va="center",
                    ha="left" if v > 0 else "right", fontsize=9, color="#52514e")


def histograma(ax, datos, titulo=None, color=AZUL, **kwargs) -> None:
    """Histograma con el estilo del proyecto (kwargs pasan a ``ax.hist``)."""
    ax.hist(datos, color=color, edgecolor="#fcfcfb", linewidth=0.8, **kwargs)
    if titulo:
        ax.set_title(titulo)
    ax.grid(axis="x", visible=False)
