# Saber 11: analítica de datos (preparación y modelamiento)

Proyecto sobre el dataset **Resultados únicos Saber 11**. Objetivo: predecir `PUNT_GLOBAL` a partir del colegio, el estudiante y su entorno familiar.

```
Proyecto-Analitica-de-Datos/
├── data/
│   ├── raw/             CSV original (saber11.csv)
│   ├── modificado/      dataset limpio (saber11_limpio.csv)
│   ├── entrenamiento/   train.csv y train_escalado.csv
│   └── test/            test.csv y test_escalado.csv
├── cuadernillos/        01_EDA, 02_Preparacion_Datos, ...
├── modelamientos/       un cuadernillo por modelo de ML
├── src/                 funciones reutilizables (preparacion.py, graficos.py)
├── despliegue/          interfaz para usar el modelo
└── presentacion/        presentación ejecutiva (sin código)
```

## Cómo reproducir

1. Clonar el repositorio e instalar dependencias: `pip install -r requirements.txt`.
2. Abrir los cuadernillos desde la carpeta `cuadernillos/`.
3. Ejecutar `02_Preparacion_Datos.ipynb`: genera `data/modificado/`, `data/entrenamiento/` y `data/test/` a partir de `data/raw/saber11.csv`.

Los archivos generados no se suben al repositorio (ver `.gitignore`) porque se recrean ejecutando el cuadernillo.
