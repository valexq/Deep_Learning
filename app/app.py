import streamlit as st
import numpy as np
import os
import json
import tensorflow as tf

# Configuración de la página
st.set_page_config(
    page_title="Reconocimiento de actividades humanas",
    page_icon="🏃",
    layout="wide"
)

# Nombres de las actividades
NOMBRES_CLASES = [
    'WALKING',
    'WALKING_UPSTAIRS',
    'WALKING_DOWNSTAIRS',
    'SITTING',
    'STANDING',
    'LAYING'
]

EMOJIS_ACTIVIDAD = {
    'WALKING': '🚶',
    'WALKING_UPSTAIRS': '🧗',
    'WALKING_DOWNSTAIRS': '⬇️',
    'SITTING': '🪑',
    'STANDING': '🧍',
    'LAYING': '🛏️'
}

# Directorios (relativos al archivo app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')


@st.cache_resource
def cargar_modelo(nombre_modelo):
    """Carga el modelo guardado."""
    rutas = {
        'MLP': os.path.join(MODELS_DIR, 'modelo_mlp.keras'),
        'CNN': os.path.join(MODELS_DIR, 'modelo_cnn.keras'),
        'RNN': os.path.join(MODELS_DIR, 'modelo_rnn.keras'),
        'LSTM': os.path.join(MODELS_DIR, 'modelo_lstm.keras'),
        'GRU': os.path.join(MODELS_DIR, 'modelo_gru.keras'),
        'Transformer': os.path.join(MODELS_DIR, 'modelo_transformer.keras'),
    }
    ruta = rutas.get(nombre_modelo)
    if ruta and os.path.exists(ruta):
        return tf.keras.models.load_model(ruta)
    return None


@st.cache_data
def cargar_datos_prueba():
    """Carga muestras del conjunto de prueba."""
    if not os.path.exists(DATA_DIR):
        return None, None, None, None
    X_seq = np.load(os.path.join(DATA_DIR, 'X_test_seq.npy'))
    X_tab = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_int = np.load(os.path.join(DATA_DIR, 'y_test_int.npy'))
    return X_seq, X_tab, y_int


@st.cache_data
def cargar_metricas():
    """Carga las métricas guardadas de todos los modelos."""
    nombres = ['mlp', 'cnn', 'rnn', 'lstm', 'gru', 'transformer']
    metricas = {}
    for nombre in nombres:
        ruta = os.path.join(MODELS_DIR, f'metricas_{nombre}.json')
        if os.path.exists(ruta):
            with open(ruta, 'r') as f:
                datos = json.load(f)
                metricas[datos['modelo']] = datos
    return metricas


# Título principal
st.title("🏃 Sistema de reconocimiento de actividades humanas")

# Barra lateral
st.sidebar.header("Configuración")

modelo_seleccionado = st.sidebar.selectbox(
    "Selecciona el modelo:",
    ['MLP', 'CNN', 'RNN', 'LSTM', 'GRU', 'Transformer']
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Sobre el proyecto**")
st.sidebar.info(
    "Este sistema usa modelos de aprendizaje profundo para reconocer "
    "actividades humanas a partir de datos de acelerómetro y giroscopio "
    "de un smartphone."
)

# Pestañas principales
tab1, tab2, tab3 = st.tabs(["🔍 Predicción", "📊 Métricas de Modelos", "ℹ️ Información"])

# ===================== TAB 1: PREDICCIÓN =====================
with tab1:
    st.header(f"Predicción con {modelo_seleccionado}")

    # Cargar datos
    X_seq, X_tab, y_int = cargar_datos_prueba()

    if X_seq is None:
        st.warning(
            "No se encontraron datos procesados. "
            "Por favor, ejecuta primero el notebook 01_EDA_Preprocessing.ipynb"
        )
    else:
        n_muestras = len(y_int)
        indice = st.slider(
            "Selecciona un registro del dataset de prueba:",
            min_value=0, max_value=n_muestras - 1, value=0
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Actividad Real")
            actividad_real = NOMBRES_CLASES[y_int[indice]]
            emoji_real = EMOJIS_ACTIVIDAD.get(actividad_real, '❓')
            st.markdown(f"### {emoji_real} {actividad_real}")

        with col2:
            st.subheader("Predicción del modelo")

            modelo = cargar_modelo(modelo_seleccionado)
            if modelo is None:
                st.error(
                    f"Modelo {modelo_seleccionado} no encontrado. "
                    "Ejecuta el notebook correspondiente primero."
                )
            else:
                # Seleccionar datos según el modelo
                if modelo_seleccionado == 'MLP':
                    muestra = X_tab[indice:indice + 1]
                else:
                    muestra = X_seq[indice:indice + 1]

                probs = modelo.predict(muestra, verbose=0)[0]
                clase_pred = np.argmax(probs)
                actividad_pred = NOMBRES_CLASES[clase_pred]
                confianza = probs[clase_pred]
                emoji_pred = EMOJIS_ACTIVIDAD.get(actividad_pred, '❓')

                st.markdown(f"### {emoji_pred} {actividad_pred}")

                if actividad_real == actividad_pred:
                    st.success(f"✓ Predicción correcta (Confianza: {confianza:.1%})")
                else:
                    st.error(f"✗ Predicción incorrecta (Confianza: {confianza:.1%})")

        # Gráfico de probabilidades
        if modelo is not None and X_seq is not None:
            st.subheader("Probabilidades por actividad")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 3))
            colores = ['green' if i == clase_pred else 'steelblue' for i in range(6)]
            ax.bar(NOMBRES_CLASES, probs, color=colores, alpha=0.85)
            ax.set_ylabel("Probabilidad")
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=30)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ===================== TAB 2: MÉTRICAS =====================
with tab2:
    st.header("Comparación de modelos")

    metricas = cargar_metricas()

    if not metricas:
        st.warning(
            "No se encontraron métricas guardadas. "
            "Ejecuta los notebooks de modelos primero."
        )
    else:
        import pandas as pd

        # Tabla de métricas
        filas = []
        for nombre, datos in metricas.items():
            filas.append({
                'Modelo': nombre,
                'Accuracy': f"{datos['accuracy']:.4f}",
                'Precision': f"{datos['precision']:.4f}",
                'Recall': f"{datos['recall']:.4f}",
                'F1-Score': f"{datos['f1']:.4f}",
                'Tiempo (s)': f"{datos['tiempo_entrenamiento']:.1f}"
            })

        df = pd.DataFrame(filas)
        st.dataframe(df.set_index('Modelo'), use_container_width=True)

        # Gráfico de barras
        st.subheader("Accuracy por modelo")
        import matplotlib.pyplot as plt

        nombres_mod = [d['modelo'] for d in metricas.values()]
        accuracies = [d['accuracy'] for d in metricas.values()]

        fig, ax = plt.subplots(figsize=(10, 4))
        colores = ['#607D8B', '#2196F3', '#9C27B0', '#4CAF50', '#FF9800', '#F44336'][:len(nombres_mod)]
        barras = ax.bar(nombres_mod, accuracies, color=colores, alpha=0.85)

        for barra, valor in zip(barras, accuracies):
            ax.text(barra.get_x() + barra.get_width() / 2,
                    barra.get_height() + 0.002,
                    f'{valor:.4f}', ha='center', va='bottom', fontsize=10)

        ax.set_ylabel('Accuracy')
        ax.set_ylim(0.7, 1.05)
        ax.set_title('Accuracy en conjunto de prueba')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ===================== TAB 3: INFORMACIÓN =====================
with tab3:
    st.header("Información del proyecto")

    st.markdown("""
    ### Descripción del Problema
    Este proyecto implementa un sistema inteligente capaz de reconocer automáticamente
    actividades humanas usando datos de sensores de un smartphone (acelerómetro y giroscopio).

    ### Dataset
    **UCI HAR Dataset** - 10,299 muestras totales, 6 clases de actividades, 30 voluntarios.

    ### Actividades Clasificadas
    | Actividad | Emoji |
    |-----------|-------|
    | WALKING | 🚶 |
    | WALKING_UPSTAIRS | 🧗 |
    | WALKING_DOWNSTAIRS | ⬇️ |
    | SITTING | 🪑 |
    | STANDING | 🧍 |
    | LAYING | 🛏️ |

    ### Modelos Implementados
    - **MLP**: Red neuronal completamente conectada (línea base)
    - **CNN**: Red convolucional 1D para patrones locales en señales
    - **RNN**: Red recurrente simple con memoria a corto plazo
    - **LSTM**: Red con compuertas para memoria a largo plazo
    - **GRU**: Versión simplificada y eficiente de la LSTM
    - **Transformer**: Arquitectura basada en mecanismo de atención
    """)
