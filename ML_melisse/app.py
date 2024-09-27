import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import pickle

# Cargar los modelos preentrenados
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Cargar los datos
df = pd.read_csv('yelp_reviews_limpio_etiquetado.csv')  # Asegúrate de que el nombre del archivo sea correcto

# Definir categorías y sus palabras clave
categories = {
    'Velocidad en la entrega': ['fast', 'quick', 'speed', 'slow', 'waiting', 'delay'],
    'Variedad de platos': ['variety', 'selection', 'options', 'choices', 'range'],
    'Comida vegetariana': ['vegetarian', 'vegan', 'plant-based', 'meat-free'],
    'Sabor': ['taste', 'flavor', 'delicious', 'tasty', 'savor'],
    'Tamaño del plato': ['portion', 'size', 'amount', 'quantity', 'enough'],
    'Presentación del plato': ['presentation', 'plating', 'look', 'visual', 'aesthetic'],
    'Postres': ['dessert', 'sweet', 'cake', 'pastry', 'ice cream', 'pudding']
}

# Función para categorizar las reseñas
def categorize_review(review):
    for category, keywords in categories.items():
        if any(keyword in review.lower() for keyword in keywords):
            return category
    return 'otros'

# Aplicar la función de categorización
df['category'] = df['review_text'].apply(categorize_review)

# Configuración del diseño
st.set_page_config(page_title="Análisis de Reseñas", layout="wide")

# Encabezado
st.title("Análisis de Reseñas del Restaurante Melisse")
st.markdown("""
    Bienvenido a la herramienta de análisis de reseñas. 
    Elige una categoría para explorar las reseñas relacionadas y obtener un análisis de sentimientos.
""")

# Opción para seleccionar categoría
selected_category = st.selectbox('Elige una categoría:', df['category'].unique())

# Filtrar reseñas por categoría seleccionada
filtered_reviews = df[df['category'] == selected_category]

# Mostrar resultados filtrados
if not filtered_reviews.empty:
    total_count_cat = filtered_reviews.shape[0]

    # Contar reseñas por sentimiento si la columna 'sentiment' existe
    if 'sentiment' in df.columns:
        positive_count_cat = filtered_reviews[filtered_reviews['sentiment'] == 'positive'].shape[0]
        negative_count_cat = filtered_reviews[filtered_reviews['sentiment'] == 'negative'].shape[0]
        neutral_count_cat = total_count_cat - (positive_count_cat + negative_count_cat)
        
        # Mostrar resultados
        st.subheader(f"Resultados para la categoría: **{selected_category}**")
        st.write(f"Total de Reseñas: **{total_count_cat}**")
        st.write(f"Reseñas Positivas: **{positive_count_cat}**")
        st.write(f"Reseñas Negativas: **{negative_count_cat}**")
        st.write(f"Reseñas Neutras: **{neutral_count_cat}**")

        # Calcular porcentajes
        positive_percentage = (positive_count_cat / total_count_cat) * 100 if total_count_cat > 0 else 0
        negative_percentage = (negative_count_cat / total_count_cat) * 100 if total_count_cat > 0 else 0

        # Mostrar porcentajes
        st.markdown(f"**Porcentaje de Reseñas Positivas:** {positive_percentage:.2f}%")
        st.markdown(f"**Porcentaje de Reseñas Negativas:** {negative_percentage:.2f}%")

        # Graficar los porcentajes
        plt.figure(figsize=(10, 6))
        plt.bar(['Positivas', 'Negativas'], [positive_percentage, negative_percentage], color=['#4CAF50', '#F44336'], alpha=0.7)
        plt.title(f'Porcentaje de Reseñas en {selected_category}', fontsize=20)
        plt.ylabel('Porcentaje (%)', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(plt)

    else:
        st.warning("No hay datos de sentimiento disponibles en las reseñas.")

else:
    st.error("No hay reseñas disponibles en esta categoría.")

# --- Sección adicional: Predecir sentimiento usando el modelo de machine learning ---

st.subheader("Predicción de Sentimientos usando Machine Learning")

# Extraer las reseñas y transformarlas usando el vectorizador
reviews_to_predict = filtered_reviews['review_text'].values
X_reviews = vectorizer.transform(reviews_to_predict)

# Predecir los sentimientos usando el modelo
predictions = model.predict(X_reviews)

# Agregar las predicciones a un DataFrame temporal para mostrarlas
predicted_sentiments = pd.DataFrame({
    'Reseña': reviews_to_predict,
    'Sentimiento Predicho': predictions
})

# Mostrar las predicciones
st.write(predicted_sentiments)

# Pie de página
st.markdown("""
    ---
    **Nota:** Este análisis está diseñado para ayudarte a comprender mejor las opiniones de los clientes. ¡Explora y descubre más!
""")
