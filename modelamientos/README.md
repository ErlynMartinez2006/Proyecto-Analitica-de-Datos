# Modelamientos

Un cuadernillo por modelo o familia de modelos, numerados a partir de `01`:

* `01_Regresion_Lineal.ipynb`
* `02_Random_Forest.ipynb`
* ...

Cada uno carga `data/entrenamiento/` y `data/test/`, entrena, evalúa y guarda el modelo en esta carpeta (por ejemplo, `modelo_rf.joblib`).
Los modelos con regresión lineal, KNN o redes usan los archivos `*_escalado.csv`; los de árboles, los `train.csv` y `test.csv`.
