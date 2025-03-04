from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import pandas as pd
import matplotlib.pyplot as plt
import io

app = FastAPI()

# 🔹 URL del dataset
URL = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"

@app.get("/get-iris")
def get_iris():
    """Devuelve el dataset de Iris en formato JSON"""
    iris = pd.read_csv(URL)
    return iris.to_dict(orient="records")  # ✅ Convertir DataFrame a JSON


@app.get("/plot-iris")
def plot_iris():
    """Genera un gráfico de dispersión de Iris y lo devuelve como imagen"""
    iris = pd.read_csv(URL)

    plt.figure(figsize=(6, 4))
    plt.scatter(iris['sepal_length'], iris['sepal_width'], alpha=0.7)
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.title("Iris Sepal Length vs Width")

    # ✅ Guardar imagen en memoria sin escribir en disco
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")
