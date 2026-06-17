import streamlit as st
import tensorflow as tf
import numpy as np
import time

st.set_page_config(
    page_title="DeepShield - Deepfake Detection",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>

.stApp {
    background-color: #f8fafc;
}

.block-container {
    max-width: 1100px !important;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.main-title {
    text-align: center;
    font-size: 2.5rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 0rem;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 1rem;
    margin-bottom: 2rem;
}

.result-box {
    background: white;
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    padding: 20px;
    margin-top: 10px;
}

.real-box {
    border-left: 6px solid #22c55e;
}

.fake-box {
    border-left: 6px solid #ef4444;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 0.9rem;
    margin-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model("deepfake_modelv2.keras")

try:
    model = load_trained_model()
    model_status = "🟢 Model Loaded Successfully"
except Exception:
    model = None
    model_status = "🔴 Model File Not Found"

st.markdown("""
<div class="main-title">
DeepShield: Deepfake Image Detection using EfficientNetB3
</div>
""", unsafe_allow_html=True)

st.info(model_status)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model", "EfficientNetB3")

with col2:
    st.metric("Input Size", "300 × 300")

with col3:
    st.metric("ROC-AUC", "0.9334")

with col4:
    st.metric("Framework", "TensorFlow")

st.divider()

st.subheader("Upload Image for Analysis")

st.caption(
    "Supported formats: JPG, JPEG and PNG. "
    "The uploaded image will be analyzed by the trained deepfake detection model."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and model is not None:

    image = tf.keras.utils.load_img(
        uploaded_file,
        target_size=(300, 300)
    )

    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)

    col_img, col_result = st.columns([1, 1])

    with col_img:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with col_result:

        st.subheader("Classification Result")

        with st.spinner("Analyzing image..."):

            start_time = time.time()

            prediction = model.predict(
                img_array,
                verbose=0
            )

            inference_time = time.time() - start_time

        score = float(prediction[0][0])

        if score < 0.5:

            confidence = 1 - score

            st.markdown("""
            <div class="result-box real-box">
            <h3>Prediction: Authentic Image</h3>
            </div>
            """, unsafe_allow_html=True)

            st.success(
                "The model classified the uploaded image as authentic. "
                "Visual patterns and feature representations extracted from "
                "the image are consistent with genuine photographic content."
            )

        else:

            confidence = score

            st.markdown("""
            <div class="result-box fake-box">
            <h3>Prediction: Deepfake Image</h3>
            </div>
            """, unsafe_allow_html=True)

            st.error(
                "The model classified the uploaded image as a deepfake. "
                "Feature representations extracted during inference indicate "
                "characteristics commonly associated with synthetically "
                "generated or manipulated facial imagery."
            )

        st.write("### Classification Confidence")
        st.write(f"**{confidence:.2%}**")
        st.progress(float(confidence))

        st.write(f"**Inference Time:** {inference_time:.3f} seconds")

        tab1, tab2 = st.tabs(
            ["Model Output", "System Information"]
        )

        with tab1:

            st.code(
f"""Sigmoid Probability : {score:.8f}

Interpretation:
Values closer to 1 indicate Deepfake.
Values closer to 0 indicate Authentic.
"""
            )

        with tab2:

            st.code(
f"""Input Shape      : (1, 300, 300, 3)
Inference Time  : {inference_time:.3f} sec
Model           : EfficientNetB3
Framework       : TensorFlow / Keras
"""
            )

elif uploaded_file is not None and model is None:

    st.error(
        "Model could not be loaded. Please ensure "
        "'deepfake_modelv2.keras' exists in the project directory."
    )

else:

    st.info(
        "System ready. Upload an image to perform deepfake detection."
    )

st.divider()

st.markdown("""
<div class="footer">

<b>DeepShield: Deepfake Image Detection using EfficientNetB3</b><br>
\

</div>
""", unsafe_allow_html=True)