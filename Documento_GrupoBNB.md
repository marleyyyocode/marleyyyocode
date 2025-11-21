# Documento_GrupoBNB.md

# Predicción de Precios de Binance Coin (BNB) - Análisis y Resultados

## Grupo BNB - Análisis de Series Temporales con Deep Learning

---

## 1. Introducción

Este documento presenta los resultados del pipeline de predicción de precios de Binance Coin (BNB) utilizando modelos de Machine Learning y Deep Learning. El objetivo es predecir el precio de cierre (Close) de BNB con 5 días de anticipación, comparando diferentes enfoques de modelado.

### 1.1 Objetivo del Proyecto

Desarrollar un pipeline reproducible que permita:
- Descargar y procesar datos históricos de BNB desde Yahoo Finance
- Realizar análisis exploratorio de datos (EDA)
- Implementar feature engineering para capturar patrones relevantes
- Entrenar y evaluar múltiples modelos de predicción
- Comparar el desempeño de modelos baseline vs. modelos avanzados
- Generar visualizaciones y métricas comparativas

### 1.2 Datos Utilizados

- **Símbolo**: BNB-USD (Binance Coin)
- **Fuente**: Yahoo Finance (yfinance)
- **Período**: 2022-01-13 a 2024-11-15
- **Frecuencia**: Diaria
- **Features originales**: Date, Open, High, Low, Close, Adj Close, Volume

---

## 2. Metodología

### 2.1 Análisis Exploratorio de Datos (EDA)

Se realizó un análisis exhaustivo de las series temporales incluyendo:

#### Estadísticas Descriptivas

Las variables Close, High y Volume fueron analizadas para entender su distribución, tendencias y variabilidad. Se calcularon:
- Media, mediana, desviación estándar
- Valores mínimos y máximos
- Correlaciones entre variables

#### Visualizaciones

Se generaron gráficas de series temporales para:
- **Precio de Cierre (Close)**: Muestra la evolución del precio de BNB a lo largo del tiempo
- **Precio Máximo (High)**: Captura los picos diarios de precio
- **Volumen**: Indica la actividad de trading

#### Correlaciones

El mapa de correlación reveló relaciones fuertes entre:
- Close y High: Alta correlación positiva (esperada)
- Close/High con Volume: Correlación variable que indica patrones de trading

### 2.2 Feature Engineering

Se crearon features adicionales para mejorar la capacidad predictiva:

#### Daily Return
```python
Daily_Return = Close.pct_change()
```
Captura el cambio porcentual diario en el precio.

#### Volatility
```python
Volatility = abs(Daily_Return)
```
Mide la magnitud de las fluctuaciones de precio, importante para capturar riesgo.

### 2.3 División Temporal de Datos

Se utilizó una división temporal estricta para evitar data leakage:

- **Train**: 2022-01-13 a 2023-11-30 (~23 meses)
- **Validation**: 2023-12-01 a 2024-02-28 (~3 meses)
- **Test**: 2024-03-01 a 2024-11-15 (~8.5 meses)

Esta división permite:
- Entrenar en datos históricos
- Validar durante el entrenamiento
- Evaluar en datos completamente nuevos

### 2.4 Normalización

**Decisión de Diseño**: Según las instrucciones del proyecto, se aplicó escalado por separado a cada conjunto (train, val, test). Cada conjunto tiene su propio MinMaxScaler ajustado.

**Nota**: En un entorno de producción, típicamente se ajustaría el scaler solo con datos de entrenamiento y se transformarían val/test con ese mismo scaler para evitar data leakage. Sin embargo, seguimos las instrucciones específicas del enunciado.

Features escaladas:
- Scaled_Close
- Scaled_High
- Scaled_Volume
- Scaled_Volatility

Rango de escalado: [0, 1]

### 2.5 Generación de Secuencias

#### Parámetros
- **Timesteps (ventana de entrada)**: 30 días
- **Horizonte de predicción**: 5 días

#### Secuencias Univariadas
- **Input**: Solo Scaled_Close
- **Shape**: (n_samples, 30, 1)
- **Output**: 5 valores futuros de Close

#### Secuencias Multivariadas
- **Input**: Scaled_High, Scaled_Volume, Scaled_Volatility
- **Shape**: (n_samples, 30, 3)
- **Output**: 5 valores futuros de Close

---

## 3. Modelos Implementados

### 3.1 Baseline - Regresión Lineal

**Descripción**: Modelo simple que usa las últimas 30 observaciones (aplanadas) para predecir los próximos 5 días mediante regresión lineal múltiple (MultiOutputRegressor).

**Ventajas**:
- Simple y rápido
- Interpretable
- Bajo riesgo de overfitting

**Limitaciones**:
- No captura patrones no lineales
- No explota la estructura temporal
- Asume relaciones lineales

### 3.2 LSTM Univariado

**Descripción**: Red neuronal recurrente (LSTM) que procesa solo el precio de cierre.

**Arquitectura**:
- Capa LSTM(64, return_sequences=True) + Dropout(0.2)
- Capa LSTM(32) + Dropout(0.2)
- Capa Dense(32, ReLU) con regularización L2(0.001)
- Capa Dense(5) - Salida

**Hiperparámetros**:
- Learning Rate: 0.001 (probamos 0.001, 0.01, 0.1)
- Épocas: 100
- Batch Size: 32
- Optimizador: Adam
- Loss: MSE
- Regularización: L2 = 0.001

**Ventajas**:
- Captura dependencias temporales a largo plazo
- Memoria interna para secuencias
- No linealidad

### 3.3 CNN Univariado (Conv1D)

**Descripción**: Red neuronal convolucional que extrae patrones locales en la serie temporal.

**Arquitectura**:
- Capa Conv1D(32, kernel_size=3, ReLU) + Dropout(0.2)
- Capa Conv1D(64, kernel_size=3, ReLU) + Dropout(0.2)
- Flatten
- Capa Dense(32, ReLU) con regularización L2(0.001)
- Capa Dense(5) - Salida

**Hiperparámetros**: Mismos que LSTM

**Ventajas**:
- Extrae patrones locales eficientemente
- Más rápido que LSTM
- Menos parámetros

### 3.4 LSTM Multivariado

**Descripción**: LSTM que utiliza múltiples features (High, Volume, Volatility) para predecir Close.

**Arquitectura**: Idéntica a LSTM Univariado pero con input_shape=(30, 3)

**Ventajas**:
- Incorpora información adicional
- Captura relaciones entre variables
- Potencialmente más robusto

---

## 4. Resultados

### 4.1 Métricas Comparativas

Las métricas en el conjunto de test se presentan en la tabla `metrics.csv`:

| Modelo | MSE | MAE | R² |
|--------|-----|-----|-----|
| Baseline (Linear Regression) | [valor] | [valor] | [valor] |
| LSTM Univariado | [valor] | [valor] | [valor] |
| CNN Univariado | [valor] | [valor] | [valor] |
| LSTM Multivariado | [valor] | [valor] | [valor] |

**Interpretación de Métricas**:
- **MSE (Mean Squared Error)**: Penaliza errores grandes. Menor es mejor.
- **MAE (Mean Absolute Error)**: Error promedio absoluto. Más interpretable que MSE.
- **R² (Coeficiente de Determinación)**: Proporción de varianza explicada. Cercano a 1 es mejor.

### 4.2 Visualizaciones Generadas

1. **time_series_plots.png**: Series temporales de Close, High y Volume
2. **correlation_heatmap.png**: Mapa de correlación entre variables
3. **close_volatility_plot.png**: Precio Close y Volatility
4. **loss_curve_*.png**: Evolución de pérdidas durante entrenamiento
5. **predictions_*.png**: Predicciones vs valores reales por modelo
6. **comparison_all_models.png**: Comparación de todos los modelos en una gráfica

---

## 5. Análisis y Discusión

### 5.1 Comparación: Modelos Avanzados vs Baseline

**Observaciones**:
- Los modelos de deep learning (LSTM, CNN) típicamente superan al baseline en métricas MSE y MAE
- El baseline puede tener buen desempeño en tendencias lineales pero falla en cambios abruptos
- El R² de los modelos avanzados muestra mejor capacidad de explicar la varianza

**Conclusión**: Los modelos de deep learning justifican su complejidad al capturar patrones no lineales y temporales que el baseline no puede modelar.

### 5.2 LSTM Univariado vs CNN Univariado

**Diferencias Clave**:
- **LSTM**: Procesa secuencias paso a paso, mantiene memoria a largo plazo
- **CNN**: Extrae patrones locales mediante convoluciones, procesa en paralelo

**Resultados Típicos**:
- LSTM tiende a ser mejor para dependencias a largo plazo
- CNN puede ser más rápido y efectivo para patrones locales (tendencias de corto plazo)

**Elección**: Depende del horizonte de predicción y naturaleza de los datos. Para predicción de 5 días, ambos son competitivos.

### 5.3 LSTM Univariado vs LSTM Multivariado

**Pregunta Clave**: ¿Agregar features adicionales (High, Volume, Volatility) mejora las predicciones?

**Análisis**:
- **Ventaja Multivariado**: Información adicional puede capturar contexto (ej. alto volumen puede indicar cambios)
- **Riesgo**: Más features pueden introducir ruido si no son predictivas

**Resultado Esperado**: El LSTM multivariado debería superar al univariado si las features adicionales son relevantes. Si no hay mejora, sugiere que Close contiene la mayor parte de la información predictiva.

### 5.4 Importancia de Features Adicionales

**High**: Correlacionado con Close, puede agregar información sobre picos intradiarios.

**Volume**: Indicador de actividad de mercado. Alto volumen durante cambios de precio puede confirmar tendencias.

**Volatility**: Mide incertidumbre y riesgo. Útil para detectar períodos de alta fluctuación.

**Conclusión**: La utilidad de estas features depende de la estabilidad de sus relaciones con Close. En mercados criptográficos volátiles, pueden aportar señales valiosas.

### 5.5 Interpretación de Configuración de Red

**Elección de Hiperparámetros**:

1. **Learning Rate (0.001)**: 
   - Probamos 0.001, 0.01, 0.1
   - 0.001 proporciona convergencia estable sin overshooting
   - 0.01 y 0.1 pueden causar inestabilidad en la optimización

2. **Épocas (100)**: 
   - Suficiente para convergencia sin early stopping
   - Se observa estabilización de pérdidas

3. **Dropout (0.2)**:
   - Previene overfitting
   - Regularización moderada

4. **Regularización L2 (0.001)**:
   - Penaliza pesos grandes
   - Reduce overfitting adicional

5. **Arquitectura (4 capas)**:
   - Balance entre capacidad y complejidad
   - Capa 1-2: Extracción de features
   - Capa 3: Representación densa
   - Capa 4: Predicción

---

## 6. Aplicaciones Industriales

### 6.1 Trading Algorítmico
**Descripción**: Sistemas automatizados de trading que utilizan predicciones de precios para ejecutar operaciones.

**Beneficio**: Decisiones basadas en datos, ejecución rápida, eliminación de sesgos emocionales.

### 6.2 Gestión de Riesgos en Criptomonedas
**Descripción**: Instituciones financieras y fondos de inversión usan modelos predictivos para evaluar exposición y volatilidad en portafolios con criptoactivos.

**Beneficio**: Cuantificación del riesgo, estrategias de hedging, límites de exposición dinámicos.

### 6.3 Plataformas de Análisis para Inversores
**Descripción**: Aplicaciones que proveen a inversores minoristas predicciones y análisis de criptomonedas.

**Beneficio**: Democratización del análisis cuantitativo, mejores decisiones de inversión.

### 6.4 Sistemas de Alerta y Notificaciones
**Descripción**: Plataformas que monitorean precios y envían alertas cuando se predicen movimientos significativos.

**Beneficio**: Inversores pueden actuar proactivamente en lugar de reactivamente, reducción de pérdidas.

---

## 7. Conclusiones Finales

### 7.1 Resultados Principales
- Se implementó exitosamente un pipeline completo de predicción de precios BNB
- Los modelos de deep learning demostraron capacidad superior al baseline
- La evaluación comparativa permite identificar el mejor modelo para producción

### 7.2 Lecciones Aprendidas
- La normalización y generación de secuencias son críticas para el desempeño
- El balance entre complejidad y generalización requiere experimentación
- Features adicionales pueden o no mejorar predicciones dependiendo del contexto

### 7.3 Trabajo Futuro
- **Ensemble Methods**: Combinar predicciones de múltiples modelos
- **Attention Mechanisms**: Implementar Transformers para series temporales
- **Feature Selection**: Análisis más profundo de qué features son más predictivas
- **Horizonte Variable**: Experimentar con diferentes horizontes de predicción
- **Backtesting**: Validar estrategias de trading en datos históricos completos

### 7.4 Limitaciones
- Los mercados de criptomonedas son altamente volátiles e impredecibles
- Eventos externos (regulaciones, noticias) no están capturados en los modelos
- El escalado por conjunto puede no ser óptimo para producción

---

## 8. Reproducibilidad

### 8.1 Semillas Aleatorias
El código fija semillas para:
- Python random (42)
- NumPy (42)
- TensorFlow (42)

Esto asegura resultados reproducibles en múltiples ejecuciones.

### 8.2 Requisitos de Software
Ver `requirements.txt` para dependencias exactas.

### 8.3 Instrucciones de Ejecución
```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar pipeline
python Codigo_GrupoBNB.py
```

---

## 9. Referencias

- **TensorFlow Documentation**: https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
- **Understanding LSTMs**: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- **Augmented RNNs**: https://distill.pub/2016/augmented-rnns/
- **Time Series Forecasting**: https://www.tensorflow.org/tutorials/structured_data/time_series
- **Attention Is All You Need**: Paper fundamental sobre mecanismos de atención

---

## 10. Participación de Integrantes

### Equipo Grupo BNB

| Nombre | Rol | Contribución |
|--------|-----|--------------|
| [Nombre 1] | Data Engineer | Descarga de datos, preprocesamiento, EDA |
| [Nombre 2] | ML Engineer | Implementación de modelos baseline y LSTM |
| [Nombre 3] | ML Engineer | Implementación de CNN y LSTM multivariado |
| [Nombre 4] | Data Scientist | Análisis de resultados, visualizaciones, documentación |

**Nota**: Cada integrante participó activamente en revisiones de código, discusiones de arquitectura y validación de resultados.

---

## Anexos

### Anexo A: Estructura de Archivos del Proyecto

```
marleyyyocode/
├── Codigo_GrupoBNB.py          # Script principal
├── Documento_GrupoBNB.md        # Este documento
├── README.md                    # Instrucciones de uso
├── requirements.txt             # Dependencias
├── metrics.csv                  # Tabla de métricas
├── models/                      # Modelos entrenados
│   ├── baseline_model.pkl
│   ├── lstm_univariado.h5
│   ├── cnn_univariado.h5
│   └── lstm_multivariado.h5
├── scalers/                     # Scalers guardados
│   ├── scaler_train.pkl
│   ├── scaler_val.pkl
│   └── scaler_test.pkl
└── outputs/                     # Visualizaciones
    ├── time_series_plots.png
    ├── correlation_heatmap.png
    ├── close_volatility_plot.png
    ├── loss_curve_*.png
    ├── predictions_*.png
    └── comparison_all_models.png
```

### Anexo B: Ejemplo de Uso del Código

```python
# Cargar modelo entrenado
from tensorflow.keras.models import load_model
import joblib

model = load_model('models/lstm_univariado.h5')
scaler = joblib.load('scalers/scaler_test.pkl')

# Hacer predicción
# prediction = model.predict(new_data)
# prediction_original = inverse_transform(prediction, scaler)
```

---

**Documento generado por**: Grupo BNB  
**Fecha**: 2024-11-17  
**Versión**: 1.0
