import requests
import json

# URL de la API
url = 'http://127.0.0.1:8000/predict_regression'

# Datos de entrada que quieres enviar
data = {
    "features": [4434 ,	16.234028 ,	-1.518749,	0,	0,	0,	0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0, 0,0	,16.126488	,16.04836]
}

# Enviar solicitud POST a la API
response = requests.post(url, json=data)

# Verificar si la solicitud fue exitosa
if response.status_code == 200:
    # Si la respuesta es exitosa, obtener los resultados
    result = response.json()
    print("Predicción de regresión:", result)
else:
    print(f"Error {response.status_code}: {response.text}")
