import streamlit as st
import numpy as np

class Neuron:
    def __init__(self, input_size, activation_function="relu"):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        self.activation_function = activation_function

    def _relu(self, x):
        return np.maximum(0, x)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _tanh(self, x):
        return np.tanh(x)

    def activate(self, x):
        if self.activation_function == "relu":
            return self._relu(x)
        elif self.activation_function == "sigmoid":
            return self._sigmoid(x)
        elif self.activation_function == "tanh":
            return self._tanh(x)
        else:
            raise ValueError("Invalid activation function")

    def run(self, input_data):
        weighted_sum = np.dot(self.weights, input_data) + self.bias
        output = self.activate(weighted_sum)
        return output

st.image('img/neurona.jpg')
st.title("Hola, Neurona!")

num_entradas = st.slider("Número de entradas/pesos de la neurona:", min_value=1, max_value=10, value=1, step=1)

col_pesos = st.columns(num_entradas)
weights = [col_pesos[i].slider(f"Seleccione el valor del peso w{i}", min_value=0.0, max_value=10.0, value=1.0, step=0.1, key=f"w{i}") for i in range(num_entradas)]

col_entradas = st.columns(num_entradas)
inputs = [col_entradas[i].number_input(f"Ingrese el valor de la entrada x{i}", step=0.1, key=f"x{i}") for i in range(num_entradas)]

bias = st.slider("Seleccione el valor del sesgo (b)", min_value=0.0, max_value=10.0, value=0.0, step=0.1, key="b")

neuron = Neuron(input_size=num_entradas)

if st.button("Calcular salida"):
    output = neuron.run(inputs)
    st.write(f"La salida de la neurona es: {output}")









