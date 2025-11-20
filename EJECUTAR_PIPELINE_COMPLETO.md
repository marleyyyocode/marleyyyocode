# 🚀 Guía de Ejecución - Pipeline Completo BNB

## 📋 Descripción

Este pipeline implementa optimización individual de hiperparámetros para cada modelo (LSTM Univariado, CNN Univariado, LSTM Multivariado) y genera proyecciones de precios 15 días hacia el futuro.

## ⏱️ Tiempo de Ejecución

**Total estimado: 30-40 minutos**

- LSTM Univariado tuning: ~10-12 min (20 experimentos)
- CNN Univariado tuning: ~10-12 min (20 experimentos)  
- LSTM Multivariado tuning: ~10-12 min (20 experimentos)
- Evaluación y proyecciones: ~3-5 min
- Visualizaciones: ~2 min

## 🔧 Requisitos Previos

```bash
# Instalar dependencias
pip install -r requirements.txt
```

## 🚀 Ejecución

### Opción 1: Script Completo Integrado (RECOMENDADO)

```bash
python run_complete_pipeline.py
```

Este script ejecuta automáticamente:
1. ✅ Preparación de datos y EDA
2. ✅ Optimización individual de hiperparámetros (3 modelos)
3. ✅ Evaluación en test set
4. ✅ Generación de proyecciones (15 días)
5. ✅ Todas las visualizaciones

### Opción 2: Ejecutar solo el módulo original

```bash
python Codigo_GrupoBNB.py
```

*(No incluye optimización individual ni proyecciones)*

## 📊 Outputs Generados

### Modelos Optimizados
```
models/
├── baseline_model_optimized.pkl
├── lstm_univariado_optimized.h5
├── cnn_univariado_optimized.h5
└── lstm_multivariado_optimized.h5
```

### Visualizaciones de Tuning (10 paneles cada una)
```
outputs/
├── tuning_lstm_univariado.png      # Análisis de hiperparámetros LSTM
├── tuning_cnn_univariado.png       # Análisis de hiperparámetros CNN
└── tuning_lstm_multivariado.png    # Análisis de hiperparámetros LSTM Multi
```

**Cada visualización incluye:**
- Learning Rate comparación
- Arquitectura (capas/filtros) comparación
- Función de Activación
- Dropout Rate
- Batch Size
- Optimizador
- Función de Pérdida
- Regularización L2
- Top 10 mejores configuraciones
- Resumen de mejor configuración (panel verde)

### Visualizaciones de Proyecciones
```
outputs/
└── projections_all_models.png      # 6 paneles:
                                     #   - 4 modelos individuales
                                     #   - 1 comparación conjunta
                                     #   - 1 panel de info
```

### Datos
```
metrics_optimized.csv          # Métricas finales (MSE, MAE, R²)
projections_15_days.csv        # Proyecciones de los 4 modelos
```

## �� Hiperparámetros Optimizados

### LSTM Univariado (20 experimentos)
- **Learning rates**: 0.001, 0.005, 0.01
- **Arquitecturas**: 
  - 2 capas: [64, 32]
  - 3 capas: [128, 64, 32]
  - 4 capas: [256, 128, 64, 32]
- **Activaciones**: relu, tanh
- **Dropout**: 0.1, 0.2, 0.3
- **Batch sizes**: 16, 32
- **Optimizadores**: Adam, RMSprop
- **Loss**: MSE, MAE
- **L2 Reg**: 0.0, 0.001

### CNN Univariado (20 experimentos)
- **Learning rates**: 0.0001, 0.0005, 0.001
- **Arquitecturas Conv**:
  - 2 capas: [32, 64]
  - 2 capas: [64, 128]
  - 2 capas: [128, 256]
- **Kernel sizes**: 3, 5
- **Activaciones**: relu, elu
- **Dropout**: 0.1, 0.2, 0.3
- **Batch sizes**: 16, 32
- **Optimizadores**: Adam, RMSprop
- **Loss**: MSE, MAE
- **L2 Reg**: 0.0, 0.001

### LSTM Multivariado (20 experimentos)
*(Similar a LSTM Univariado, optimizado para 3 features: High, Volume, Volatility)*

## 🎯 Características Clave

✅ **Optimización Individual por Modelo**
- Cada modelo encuentra sus hiperparámetros óptimos
- Incluye hiperparámetros arquitectónicos (capas, unidades, filtros)
- Total: ~60 experimentos (20 por modelo)

✅ **Proyecciones Multi-Paso**
- 15 días hacia el futuro
- Proyección iterativa para todos los modelos
- Visualización comparativa

✅ **Sin Early Stopping**
- Como se requirió en las especificaciones
- Épocas fijas (80) para cada experimento

✅ **Reproducibilidad**
- Seed fija (SEED=42) en Python, NumPy, TensorFlow
- Resultados reproducibles

## 💡 Interpretación de Resultados

### Visualizaciones de Tuning

**Panel verde = Mejor valor** para cada hiperparámetro

Cada gráfico muestra:
- Valor promedio de validation loss para cada configuración
- Mejor valor resaltado en verde
- Permite identificar qué hiperparámetros tienen mayor impacto

### Métricas

**Baseline (Regresión Lineal)** = Referencia
- Si deep learning NO supera al baseline → Datos tienen patrones lineales
- Si deep learning supera → Capturó patrones no-lineales complejos

**R² (Coeficiente de Determinación)**:
- 1.0 = Predicción perfecta
- 0.9+ = Excelente
- 0.8-0.9 = Bueno
- <0.8 = Regular/Necesita mejora

**MSE y MAE**:
- Menor es mejor
- MSE penaliza errores grandes más fuertemente
- MAE es más interpretable (error promedio en USD)

### Proyecciones

- **Línea azul**: Datos reales (test set)
- **Líneas de colores**: Proyecciones de cada modelo
- **Línea vertical punteada**: Inicio de proyección

**Divergencia entre modelos** = Incertidumbre sobre precios futuros

## 🐛 Troubleshooting

### Error: `ModuleNotFoundError`
```bash
pip install -r requirements.txt
```

### Error: `No module named 'enhanced_additions'`
Asegúrate de estar en el directorio correcto:
```bash
cd /path/to/marleyyyocode/
python run_complete_pipeline.py
```

### Proceso muy lento
- Normal para 60 experimentos (~30-40 min)
- Puedes monitorear el progreso en la consola
- Cada experimento muestra su número y validation loss

### Out of Memory
- Reduce batch size en experimentos
- Ejecuta en máquina con más RAM
- O ejecuta modelos por separado

## 📝 Notas Importantes

1. **Datos Sintéticos**: Si yfinance API falla, el código usa datos sintéticos automáticamente. Los resultados siguen siendo válidos para demostrar el pipeline.

2. **GPU vs CPU**: El código funciona en ambos. GPU es más rápido pero no es necesario.

3. **Resultados Variables**: Aunque usamos seed=42, pequeñas variaciones pueden ocurrir debido a operaciones no-deterministas en TensorFlow.

4. **Almacenamiento**: Asegúrate de tener ~100MB libres para modelos y visualizaciones.

## 📚 Estructura del Código

```
marleyyyocode/
├── Codigo_GrupoBNB.py              # Pipeline base original
├── enhanced_additions.py           # Funciones de tuning + proyecciones
├── run_complete_pipeline.py        # Script integrado ejecutable
├── EJECUTAR_PIPELINE_COMPLETO.md   # Esta guía
├── requirements.txt                # Dependencias
├── models/                         # Modelos entrenados (generado)
├── outputs/                        # Visualizaciones (generado)
└── scalers/                        # Scalers guardados (generado)
```

## 🎓 Referencias

- **LSTM**: https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
- **Conv1D**: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D
- **Time Series**: https://www.tensorflow.org/tutorials/structured_data/time_series

## ✅ Checklist de Ejecución

- [ ] Dependencias instaladas (`pip install -r requirements.txt`)
- [ ] Directorio correcto (donde está `run_complete_pipeline.py`)
- [ ] Espacio en disco suficiente (~100MB)
- [ ] Tiempo disponible (~30-40 min)
- [ ] Ejecutar: `python run_complete_pipeline.py`
- [ ] Revisar outputs en `models/` y `outputs/`
- [ ] Verificar métricas en `metrics_optimized.csv`
- [ ] Analizar proyecciones en `projections_15_days.csv`

---

**¡Listo para ejecutar!** 🚀

Para dudas o problemas, revisar la sección de Troubleshooting arriba.
