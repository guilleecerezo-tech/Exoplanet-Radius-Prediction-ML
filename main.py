import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#Importamos el dataset
dataset = pd.read_csv('planet_data.csv', comment='#')
X = dataset[['pl_massj','pl_eqt']].values
y = dataset[['pl_radj']].values

#LLenamos datos vacios mediante sklearn
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)

#Dividimos el dataset para una mayor lectura
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Entrenamos el modelo con Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators= 500, max_depth=10, random_state=0)
regressor.fit(X_train,y_train)

# Predecimos los resultados del conjunto de test
y_pred = regressor.predict(X_test)

#MÉTRICAS DE ERROR 
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("-" * 30)
print("MÉTRICAS DEL MODELO")
print("-" * 30)
print(f"Error Cuadrático Medio (MSE): {mse:.4f}")
print(f"Error Absoluto Medio (MAE): {mae:.4f}")
print(f"Coeficiente de determinación (R²): {r2:.4f}")
print("-" * 30)


# --- VISUALIZACIÓN ---
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 7))

# Dibujamos solo una vez y guardamos en 'scatter'
scatter = ax.scatter(X_test[:, 0], y_test, c=X_test[:, 1], cmap='inferno', s=50, alpha=0.7, edgecolors='w', linewidth=0.5)
cbar = plt.colorbar(scatter)
cbar.set_label('Temperatura de Equilibrio (K)')

# Referencias
ax.axvline(1, color='gray', linestyle='--', alpha=0.3)
ax.axhline(1, color='gray', linestyle='--', alpha=0.3)

ax.set_title('Predicción del tamaño de un exoplaneta', fontsize=15)
ax.set_xlabel('Masa (unidades de Júpiter)')
ax.set_ylabel('Radio (unidades de Júpiter)')
ax.set_xlim(-0.5, 15)
ax.set_ylim(-0.2, 3)
plt.show()