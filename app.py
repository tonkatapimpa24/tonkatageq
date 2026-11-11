import streamlit as st
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️")
st.title("✍️ Handwritten Digit Recognition")
st.write("Upload a handwritten digit image and AI will try to recognize it.")

# Кеширане и обучение на модела
@st.cache_resource
def load_model():
    try:
        digits = load_digits()
        X = digits.images.reshape((len(digits.images), -1)) / 16.0
        y = digits.target
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=300,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

model = load_model()

if model is None:
    st.warning("Could not load model.")
else:
    st.success("Model loaded successfully!")

# Качване на файл
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    try:
        # Преобразуване в черно-бяло и оразмеряване до 8x8
        img_gray = image.convert('L')
        img_resized = img_gray.resize((8, 8), Image.Resampling.LANCZOS)

        img_array = np.array(img_resized, dtype=np.float32)

        # Ако фонът е светъл (бял лист с тъмен химикал), инвертираме цветовете,
        # защото в набора load_digits фонът е черен (0), а цифрата е бяла (16).
        if np.mean(img_array) > 128:
            img_array = 255.0 - img_array

        # Нормализиране в диапазона [0.0, 1.0] и преобразуване във вектор (1, 64)
        img_normalized = img_array / 255.0
        img_flattened = img_normalized.reshape(1, -1)

        # Прогноза
        prediction = model.predict(img_flattened)[0]
        probabilities = model.predict_proba(img_flattened)[0]
        confidence = np.max(probabilities) * 100

        st.subheader(f"Разпозната цифра: **{prediction}**")
        st.write(f"Сигурност на модела: **{confidence:.1f}%**")

        # Визуализация на вероятностите за всички цифри (0-9)
        st.bar_chart(probabilities)

    except Exception as e:
        st.error(f"Error processing image: {e}")
