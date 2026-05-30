<<<<<<< HEAD
# Deep_Learning
=======
# Sistema de Reconocimiento de Actividades Humanas (HAR)

## Integrantes:
- Laura Escobar Rojo
- Vanessa Alfaro

## Descripción del Problema

Diseñar un sistema inteligente capaz de reconocer automáticamente actividades humanas realizadas por una persona utilizando datos provenientes de sensores móviles (acelerómetro y giroscopio), con el fin de apoyar procesos de monitoreo, salud, deporte y asistencia inteligente.

## Actividades clasificadas

| ID | Actividad |
|----|-----------|
| 1 | WALKING |
| 2 | WALKING_UPSTAIRS |
| 3 | WALKING_DOWNSTAIRS |
| 4 | SITTING |
| 5 | STANDING |
| 6 | LAYING |

## Estructura del proyecto

```
project/
│
├── data/
│   ├── UCI HAR Dataset/        # Dataset original 
│   └── processed/              # Datos preprocesados (generados por notebook 01)
│
├── notebooks/
│   ├── 01_EDA_Preprocessing.ipynb   # Análisis exploratorio y preprocesamiento
│   ├── 02_MLP.ipynb                 # Red Neuronal Profunda
│   ├── 03_CNN.ipynb                 # Red Convolucional 1D
│   ├── 04_RNN.ipynb                 # Red Recurrente Simple
│   ├── 05_LSTM.ipynb                # Long Short-Term Memory
│   ├── 06_GRU.ipynb                 # Gated Recurrent Unit
│   ├── 07_Transformer.ipynb         # Transformer
│   └── 08_Comparacion_Modelos.ipynb # Comparación final
│
├── models/                     # Modelos entrenados guardados (.keras)
│
├── app/
│   └── app.py                  # Aplicación Streamlit
│
├── reports/                    # Gráficos y reportes generados
│
├── requirements.txt
└── README.md
```

## Instalación

```bash
# 1. Clonar o descargar el proyecto
# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

## Ejecución

### Orden de ejecución de los notebooks

Los notebooks deben ejecutarse en orden. El notebook 01 genera los datos preprocesados que usan los demás:

```
01_EDA_Preprocessing.ipynb  →  genera data/processed/
02_MLP.ipynb                →  genera models/modelo_mlp.keras
03_CNN.ipynb                →  genera models/modelo_cnn.keras
04_RNN.ipynb                →  genera models/modelo_rnn.keras
05_LSTM.ipynb               →  genera models/modelo_lstm.keras
06_GRU.ipynb                →  genera models/modelo_gru.keras
07_Transformer.ipynb        →  genera models/modelo_transformer.keras
08_Comparacion_Modelos.ipynb → análisis comparativo final
```

Para ejecutar los notebooks:
```bash
jupyter notebook
```

### Ejecutar la aplicación Streamlit

```bash
cd app
streamlit run app.py
```

La aplicación se abrirá en el navegador en `http://localhost:8501`

## Modelos implementados

| Modelo | Datos de entrada | Arquitectura |
|--------|-----------------|--------------|
| MLP | Tabular (561 feat.) | 3 capas densas + Dropout + BatchNorm |
| CNN | Señal cruda (128×9) | 3 Conv1D + MaxPooling + Dense |
| RNN | Señal cruda (128×9) | 2 SimpleRNN + Dense |
| LSTM | Señal cruda (128×9) | 2 LSTM + Dense |
| GRU | Señal cruda (128×9) | 2 GRU + Dense |
| Transformer | Señal cruda (128×9) | Proyección + 2 bloques Transformer |

## Resultados obtenidos

Los resultados exactos dependen de la ejecución, pero en general se espera:

| Modelo | Accuracy esperada |
|--------|------------------|
| MLP | ~93-95% |
| CNN | ~92-95% |
| RNN | ~88-91% |
| LSTM | ~92-95% |
| GRU | ~92-95% |
| Transformer | ~91-94% |

## Métricas de evaluación

Todos los modelos se evalúan con:
- **Accuracy**: proporción de predicciones correctas
- **Precision**: de las predicciones positivas, cuántas eran correctas
- **Recall**: de los casos positivos reales, cuántos fueron detectados
- **F1-Score**: media armónica de precision y recall
- **Matriz de confusión**: distribución de errores por clase

## Conclusiones

- Los modelos recurrentes (LSTM y GRU) son naturalmente adecuados para datos de sensores por su capacidad de capturar dependencias temporales.
- La GRU ofrece el mejor equilibrio entre desempeño y eficiencia computacional.
- El MLP, aunque simple, obtiene buenos resultados gracias a las 561 características ya extraídas.
- El Transformer es competitivo y tiene la ventaja de procesar las secuencias en paralelo.
- Todos los modelos superan el 88% de accuracy, lo que indica que el problema está bien definido y los datos son de buena calidad.

## Recomendaciones futuras

1. Explorar modelos híbridos CNN + LSTM.
2. Aplicar búsqueda automática de hiperparámetros (Keras Tuner).
3. Evaluar técnicas de data augmentation para señales de sensores.
4. Extender el sistema a más actividades con datasets más grandes.
5. Desplegar el modelo en un dispositivo móvil para inferencia en tiempo real.
>>>>>>> bd6b5b3 (Second commit)
