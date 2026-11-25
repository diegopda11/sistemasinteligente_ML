# 1. Importar librerías necesarias
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 2. Cargar el archivo CSV y eliminar filas con valores nulos
df = pd.read_csv("sensores_planta.csv")
df = df.dropna()

# 3. Separar las 4 variables de entrada (X) y la variable objetivo (y)
X = df[["temperatura_c", "vibracion_mms", "horas_operacion", "presion_psi"]]
y = df["estado"]

# 4. Dividir los datos en entrenamiento (80%) y prueba (20%) manteniendo proporción de clases
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Instanciar y entrenar el modelo RandomForestClassifier
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# 6. Evaluar el modelo e imprimir la precisión (Accuracy)
y_pred = modelo.predict(X_test)
precision = accuracy_score(y_test, y_pred)
print(f"🎯 Precisión del modelo (Accuracy): {precision * 100:.2f}%")

# 7. Exportar el modelo entrenado a archivo .pkl
joblib.dump(modelo, "modelo_mantenimiento.pkl")
print("✅ Modelo exportado correctamente en 'modelo_mantenimiento.pkl'")

