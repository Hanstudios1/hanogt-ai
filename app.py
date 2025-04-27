import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class RegressionModel:
    def __init__(self, degree=1):
        self.degree = degree
        if degree == 1:
            self.model = LinearRegression()
        else:
            self.model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        self.is_trained = False

    def train(self, X, y, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.model.fit(self.X_train, self.y_train)
        self.is_trained = True

    def evaluate(self):
        if not self.is_trained:
            raise Exception("Model önce eğitilmelidir!")
        
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        metrics = {
            "Train R2": r2_score(self.y_train, y_train_pred),
            "Test R2": r2_score(self.y_test, y_test_pred),
            "Test MAE": mean_absolute_error(self.y_test, y_test_pred),
            "Test MSE": mean_squared_error(self.y_test, y_test_pred)
        }
        return metrics

    def predict(self, X_new):
        if not self.is_trained:
            raise Exception("Model önce eğitilmelidir!")
        return self.model.predict(X_new)

    def plot(self):
        if not self.is_trained:
            raise Exception("Model önce eğitilmelidir!")

        plt.figure(figsize=(8,6))
        plt.scatter(self.X_train, self.y_train, color='blue', label='Eğitim Verisi')
        plt.scatter(self.X_test, self.y_test, color='green', label='Test Verisi')

        X_range = np.linspace(self.X_train.min()-1, self.X_train.max()+1, 100).reshape(-1, 1)
        y_range_pred = self.model.predict(X_range)

        plt.plot(X_range, y_range_pred, color='red', linewidth=2, label='Model Tahmini')

        plt.xlabel('X Değeri')
        plt.ylabel('Y Değeri')
        plt.title(f'Derece {self.degree} Regresyon Modeli')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

# Streamlit Arayüzü
st.title("Hanogt AI - Basit Regresyon Modeli")

st.sidebar.header("Ayarlar")
degree = st.sidebar.slider("Model Derecesi (Doğrusal=1, Polinomial>1)", 1, 5, 1)
n_samples = st.sidebar.slider("Veri Sayısı", 20, 500, 100)
noise = st.sidebar.slider("Gürültü (Noise)", 0, 50, 10)

st.write("## Veri Seti ve Model Eğitimi")
np.random.seed(42)
X = np.random.rand(n_samples, 1) * 10
y = 3 * (X.squeeze()**2) + 2 * X.squeeze() + np.random.randn(n_samples) * noise

model = RegressionModel(degree=degree)
model.train(X, y)

metrics = model.evaluate()

st.write("### Model Performansı")
for metric, value in metrics.items():
    st.write(f"**{metric}**: {value:.4f}")

st.write("### Model Grafiği")
model.plot()

st.write("## Yeni Tahminler Yap")
input_value = st.number_input("Bir X değeri gir:", min_value=0.0, max_value=20.0, value=5.0)
if st.button("Tahmin Yap"):
    prediction = model.predict(np.array([[input_value]]))
    st.success(f"{input_value:.2f} için tahmin: {prediction[0]:.2f}")