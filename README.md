# Análisis de Percepción de Amenaza de la IA en Desarrolladores

## Librerías Utilizadas

Este proyecto hace uso de las siguientes librerías de Python para el análisis de datos, visualización y modelado:

*   `pandas`: Manipulación y análisis de datos.
*   `matplotlib`: Creación de visualizaciones estáticas e interactivas.
*   `seaborn`: Visualización de datos estadísticos basada en `matplotlib`.
*   `sklearn` (Scikit-learn): Herramientas de machine learning, incluyendo preprocesamiento y modelos.
*   `numpy`: Soporte para arrays y operaciones numéricas de alto nivel.
*   `xgboost`: Implementación optimizada del algoritmo Gradient Boosting para modelos de árboles.
*   `lightgbm`: Framework de Gradient Boosting que utiliza técnicas basadas en árboles de decisión.

## Motivación del Proyecto

En la era actual, la Inteligencia Artificial (IA) está transformando rápidamente diversas industrias, y el sector del desarrollo de software no es la excepción. Sin embargo, la introducción de nuevas tecnologías a menudo genera tanto entusiasmo como preocupación, especialmente en lo que respecta al impacto en la fuerza laboral. Este proyecto se enfoca en analizar la percepción de los desarrolladores de software sobre la IA como una amenaza, explorando qué factores demográficos, profesionales y tecnológicos pueden influir en esta percepción.

El objetivo principal es identificar patrones y construir modelos predictivos que permitan comprender mejor las variables asociadas con una percepción de amenaza de la IA. Este análisis es crucial para la comunidad de desarrolladores y las organizaciones, ya que puede informar estrategias para mitigar temores, fomentar la adaptación tecnológica y asegurar una transición más fluida hacia un futuro con mayor integración de la IA en el ámbito del desarrollo de software. Al entender quiénes y por qué perciben a la IA como una amenaza, podemos abordar estas preocupaciones de manera más efectiva y promover una adopción positiva de la IA.

## Descripción de los Archivos

Este proyecto consta de los siguientes archivos:

*   `notebook.ipynb`: El cuaderno de Jupyter que contiene todo el código fuente para el análisis de datos, preprocesamiento, modelado, evaluación de modelos y la predicción del escenario hipotético.
*   `survey_results_public.csv`: El conjunto de datos original utilizado para este análisis. Este archivo contiene los resultados de una encuesta pública que aborda diversas características de los desarrolladores y su entorno laboral, incluyendo su percepción sobre la Inteligencia Artificial.

## Resultados del Análisis

### Resumen de Preprocesamiento y Selección de Características

El proceso de preparación de los datos fue crucial para el modelado, y se realizó en varias etapas:

1.  **Carga y Limpieza Inicial:** El conjunto de datos `survey_results_public.csv` fue cargado y la variable objetivo `AIThreat` fue limpiada. Se eliminaron filas con valores nulos para `AIThreat` y las respuestas 'I'm not sure', binarizando la variable a `0` (No) y `1` (Yes).
2.  **Manejo de Valores Faltantes:** Se identificaron y eliminaron columnas con más del 70% de valores faltantes para reducir el ruido y la complejidad.
3.  **Codificación de Características Categóricas:** Las variables categóricas con menos de 5 valores únicos fueron transformadas usando `OneHotEncoder` (con `drop='first'`) para evitar multicolinealidad. Los valores nulos en estas columnas, resultantes de la codificación de categorías 'NaN', se imputaron con `0`.
4.  **Imputación de Variables Numéricas:** Todos los valores faltantes en las columnas numéricas restantes fueron imputados con `0` para asegurar la completitud del dataset.
5.  **Análisis de Information Value (IV):** Se calculó el IV para cada característica. El IV es una medida de la capacidad predictiva de una variable y fue clave en la selección final de características.
6.  **Selección de Características por Correlación:** Se realizó un análisis de correlación sobre el conjunto de entrenamiento. Para pares de características con una correlación absoluta superior a 0.6, se eliminó aquella con el menor valor de IV. Este proceso se aplicó de manera iterativa hasta reducir la multicolinealidad, resultando en un conjunto final de 24 características para el modelado.

### Tabla de Métricas de Rendimiento

| Métrica        | Regresión Logística | XGBoost Classifier | LightGBM Classifier |
|:---------------|:--------------------|:-------------------|:--------------------|
| **Accuracy**   | 0.1579              | 0.8407             | 0.8434              |
| **Precision**  | 0.1566              | 0.3441             | 0.5000              |
| **Recall**     | 0.9988              | 0.0190             | 0.0048              |
| **F1-Score**   | 0.2708              | 0.0361             | 0.0094              |
| **ROC AUC**    | 0.5000              | 0.5915             | 0.6010              |

### Análisis de Rendimiento de los Modelos y Selección

El problema de predicción de 'AIThreat' es un problema de clasificación binaria con un significativo desequilibrio de clases, donde la clase positiva ('Yes' - percepción de amenaza) es minoritaria. En tales escenarios, la `Accuracy` por sí sola puede ser engañosa, ya que un modelo podría lograr una alta precisión simplemente clasificando la mayoría de las instancias como la clase mayoritaria.

#### Regresión Logística
A pesar de tener un `Recall` relativamente moderado (0.988) en comparación con su `Precision` (0.1566), el modelo de Regresión Logística mostró un `ROC AUC` de 0.5000. Sin embargo, su `F1-Score` de 0.2708 indica que no logra un equilibrio efectivo entre la detección de verdaderos positivos y el control de falsos positivos.

#### XGBoost Classifier
El modelo XGBoost logró una `Accuracy` de 0.8407, lo que sugiere una buena clasificación general. Sin embargo, su `Precision` fue de 0.3441 y su `Recall` de 0.0190, resultando en un `F1-Score` de 0.0361. Un `ROC AUC` de 0.5915 indica una capacidad discriminatoria ligeramente superior a la aleatoria. La principal debilidad de este modelo es su muy bajo `Recall`, lo que significa que es ineficaz para identificar la mayoría de las instancias de la clase positiva.

#### LightGBM Classifier
LightGBM mostró una `Accuracy` de 0.8434. Aunque presentó una `Precision` de 0.5000 y un `Recall` de 0.0048, su `ROC AUC` fue de 0.6010. El `F1-Score` de 0.0094 refleja el desequilibrio en sus predicciones. El bajo `Recall` sugiere que el modelo es excesivamente conservador al predecir la clase positiva, fallando en identificar casi todas las amenazas de IA reales.

### Conclusión y Modelo Seleccionado

Ninguno de los modelos en su estado actual (Regresión Logística, XGBoost, LightGBM) es robusto para el problema de 'AIThreat' si el objetivo primordial es identificar la clase minoritaria de manera efectiva. Los modelos de *boosting* (XGBoost y LightGBM) están fuertemente sesgados hacia la clase mayoritaria ('No'), como lo demuestra su alto `Accuracy` pero muy bajo `Recall` y `F1-Score` para la clase minoritaria.

Para el propósito de **servir de base para la predicción de un nuevo escenario**, y buscando el modelo con la **mayor capacidad discriminatoria general**, el **LightGBM Classifier** fue seleccionado. A pesar de que sus métricas de `Precision`, `Recall` y `F1-Score` son muy bajas en la detección de la clase positiva, su `ROC AUC` es ligeramente competitivo. Es crucial reconocer que este modelo requiere una **optimización profunda, especialmente en el manejo del desequilibrio de clases**, para mejorar significativamente su capacidad de detectar la clase 'Yes' de 'AIThreat'. Se necesitarán técnicas de rebalanceo (como `SMOTE`) o ajuste de umbrales para que sea verdaderamente útil en un escenario donde el `Recall` de 'AIThreat' sea importante.

Analizando más a fondo nuestros datos, confirmamos que la gran mayoría de los desarrolladores no perciben la IA como una amenaza directa, lo cual es un indicativo positivo. Sin embargo, aquellos con mayor satisfacción laboral y mejor compensación económica muestran una menor inclinación a sentirse amenazados. La experiencia profesional también influye, revelando patrones interesantes sobre cómo los desarrolladores a lo largo de su carrera se adaptan o reaccionan a la irrupción de la IA

## Agradecimientos

Este proyecto fue desarrollado como parte de un ejercicio de análisis de datos y modelado predictivo. Agradecemos a Stack Overflow por la recopilación de datos a través de su encuesta pública, que hizo posible este estudio.
