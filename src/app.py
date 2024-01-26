import streamlit as st
import numpy as np

class Neuron:
    def __init__(self, input_size, activation_function="relu"):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        self.activation_function = activation_function

    def _relu(self, x):
        return np.maximum(0, x)

    def _sigmoide(self, x):
        return 1 / (1 + np.exp(-x))

    def _tangente_hiperbolica(self, x):
        return np.tanh(x)

    def activate(self, x):
        if self.activation_function == "relu":
            return self._relu(x)
        elif self.activation_function == "sigmoide":
            return self._sigmoide(x)
        elif self.activation_function == "tangente_hiperbolica":
            return self._tangente_hiperbolica(x)
        else:
            raise ValueError("Invalid activation function")

    def run(self, input_data):
        weighted_sum = np.dot(self.weights, input_data) + self.bias
        output = self.activate(weighted_sum)
        return output

st.image('img/neurona.jpg')
st.title("Hola, Neurona!")

# Pestaña para configurar la neurona
num_entradas = st.slider("Número de entradas/pesos de la neurona:", min_value=1, max_value=10, value=1, step=1)

col_pesos = st.columns(num_entradas)
weights = [col_pesos[i].slider(f"Seleccione el valor del peso w{i}", min_value=0.0, max_value=10.0, value=1.0, step=0.1, key=f"w{i}") for i in range(num_entradas)]

# Pestaña para configurar el sesgo
with st.expander("Configuración del sesgo"):
    bias = st.slider("Seleccione el valor del sesgo (b)", min_value=0.0, max_value=10.0, value=0.0, step=0.1, key="b")

# Pestaña para configurar las entradas
with st.expander("Configuración de las entradas"):
    col_entradas = st.columns(num_entradas)
    inputs = [col_entradas[i].number_input(f"Ingrese el valor de la entrada x{i}", step=0.1, key=f"x{i}") for i in range(num_entradas)]

# Pestaña para elegir la función de activación
with st.expander("Configuración de la función de activación"):
    st.write("Seleccione la función de activación:")
    activation_function = st.radio("", ["relu", "sigmoide", "tangente_hiperbolica"], key="activation_function")
    st.write(f"Función de activación actual: {activation_function}")

# Crear la neurona
neuron = Neuron(input_size=num_entradas, activation_function=activation_function)

if st.button("Calcular salida"):
    output = neuron.run(inputs)
    st.write(f"La salida de la neurona es: {output}")
