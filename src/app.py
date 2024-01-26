import streamlit as st

st.image('img/neurona.jpg')
st.title("Hola, Neurona!")

tab1, tab2, tab3 = st.tabs(["Una Variable", "Dos Variables", "Tres Variables"])

with tab1:
    st.header("Una Variable de Entrada")
    
    col_pesos = st.columns(2)
    w0 = col_pesos[0].slider("Seleccione el valor del peso w0", min_value=0.0, max_value=10.0, value=1.0, step=0.1, key="w0_una")
    
    col_entradas = st.columns(1)
    x0 = col_entradas[0].number_input("Ingrese el valor de la entrada x0", step=0.1, key="x0_una")
    
    col_sesgo = st.columns(1)
    b = col_sesgo[0].slider("Seleccione el valor del sesgo (b)", min_value=0.0, max_value=10.0, value=0.0, step=0.1, key="b_una")

    st.write("")
    if st.button("Calcular salida", key="calcular_una_variable"):
        operacion = w0 * x0 + b
        st.write(f"La salida de la neurona es: {operacion}")

with tab2:
    st.header("Dos Variables de Entrada")
    
    col_pesos = st.columns(2)
    w0 = col_pesos[0].slider("Seleccione el valor del peso w0", min_value=0.0, max_value=10.0, value=1.0, step=0.1, key="w0_dos")
    w1 = col_pesos[1].slider("Seleccione el valor del peso w1", min_value=0.0, max_value=10.0, value=1.0, step=0.1, key="w1_dos")
    
    col_entradas = st.columns(2)
    x0 = col_entradas[0].number_input("Ingrese el valor de la entrada x0", step=0.1, key="x0_dos")
    x1 = col_entradas[1].number_input("Ingrese el valor de la entrada x1", step=0.1, key="x1_dos")
    
    col_sesgo = st.columns(1)
    b = col_sesgo[0].slider("Seleccione el valor del sesgo (b)", min_value=0.0, max_value=10.0, value=0.0, step=0.1, key="b_dos")

    st.write("")
    if st.button("Calcular salida", key="calcular_dos_variables"):
        operacion = w0 * x0 + w1 * x1 + b
        st.write(f"La salida de la neurona es: {operacion}")

with tab3:

    
    col_pesos = st.columns(3)
    w0 = col_pesos[0].slider("Seleccione el valor del peso w0", min_value=0.0, max_value=10.0, value=1.0, step=0.1, key="w0_tres")
    w1 = col_pesos[1].slider("Seleccione el valor del peso w1", min_value=0.0, max_value=10.0, value=1.0, step=0.1, key="w1_tres")
    w2 = col_pesos[2].slider("Seleccione el valor del peso w2", min_value=0.0, max_value=10.0, value=1.0, step=0.1, key="w2_tres")
    
    col_entradas = st.columns(3)
    x0 = col_entradas[0].number_input("Ingrese el valor de la entrada x0", step=0.1, key="x0_tres")
    x1 = col_entradas[1].number_input("Ingrese el valor de la entrada x1", step=0.1, key="x1_tres")
    x2 = col_entradas[2].number_input("Ingrese el valor de la entrada x2", step=0.1, key="x2_tres")
    
    col_sesgo = st.columns(1)
    b = col_sesgo[0].slider("Seleccione el valor del sesgo (b)", min_value=0.0, max_value=10.0, value=0.0, step=0.1, key="b_tres")

    if st.button("Calcular salida", key="calcular_tres_variables"):
        operacion = w0 * x0 + w1 * x1 + w2 * x2 + b
        st.write(f"La salida de la neurona es:",operacion)










