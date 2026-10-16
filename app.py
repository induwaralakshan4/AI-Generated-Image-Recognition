
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib
from pathlib import Path

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI-Generated Image Recognition",
    page_icon="🔍",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("Ai_and_Real_Face_Recognize.sav")

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# Model labels
label_dict = {
    0: "Real Images",
    1: "AI-Generated Images"
}

# -----------------------------
# Example Images
# -----------------------------
real_images = [
    "real_faces/1.jpg",
    "real_faces/2.jpg",
    "real_faces/3.jpg"
]

ai_images = [
    "ai_faces/1.jpg",
    "ai_faces/2.png",
    "ai_faces/3.png"
]

# -----------------------------
# Session State
# -----------------------------
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None

if "prediction" not in st.session_state:
    st.session_state.prediction = None

# -----------------------------
# Image Processing
# -----------------------------
def process_image(image):
    if image is None:
        return "Please upload or select an image first!"

    try:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        elif not isinstance(image, Image.Image):
            image = Image.fromarray(
                np.asarray(image).astype(np.uint8)
            ).convert("RGB")

        else:
            image = image.convert("RGB")

        img_array = np.array(image)

        # Convert RGB to grayscale
        gray = cv2.cvtColor(
            img_array,
            cv2.COLOR_RGB2GRAY
        )

        # Resize to 8x8 pixels
        gray = cv2.resize(gray, (8, 8))

        # Prepare features
        features = gray.reshape(1, 64)

        # Predict
        result = model.predict(features)[0]

        return label_dict.get(
            int(result),
            f"Unknown class: {result}"
        )

    except Exception as e:
        return f"Prediction error: {e}"

# -----------------------------
# Select Example Image
# -----------------------------
def select_example(image_path):
    try:
        image = Image.open(image_path).convert("RGB")

        st.session_state.selected_image = image

        # Automatically predict the selected example
        st.session_state.prediction = process_image(image)

    except Exception as e:
        st.error(f"Could not load selected image: {e}")

# -----------------------------
# Page Heading
# -----------------------------
st.title("🔍 AI-Generated Image Recognition")

st.write(
    "Upload your own image or click an example image "
    "to automatically predict its category."
)

st.divider()

# -----------------------------
# Main Layout
# -----------------------------
left_col, right_col = st.columns([1.2, 1])

# LEFT SIDE: Prediction + Example Galleries
with left_col:

    st.subheader("Prediction Result")

    if st.session_state.prediction is not None:
        result = st.session_state.prediction

        if result == "Real Images":
            st.success(f"Prediction: {result}")

        elif result == "AI-Generated Images":
            st.warning(f"Prediction: {result}")

        else:
            st.error(result)

    else:
        st.info("Select an example or upload an image to begin.")

    st.divider()

    # Real Face Gallery
    st.subheader("Real Face Images")

    real_cols = st.columns(3)

    for i, image_path in enumerate(real_images):

        with real_cols[i]:

            if Path(image_path).is_file():

                st.image(
                    image_path,
                    use_container_width=True
                )

                if st.button(
                    "Select",
                    key=f"real_{i}",
                    use_container_width=True
                ):
                    select_example(image_path)
                    st.rerun()

            else:
                st.caption(f"Image not found: {image_path}")

    st.divider()

    # AI Face Gallery
    st.subheader("AI-Generated Face Images")

    ai_cols = st.columns(3)

    for i, image_path in enumerate(ai_images):

        with ai_cols[i]:

            if Path(image_path).is_file():

                st.image(
                    image_path,
                    use_container_width=True
                )

                if st.button(
                    "Select",
                    key=f"ai_{i}",
                    use_container_width=True
                ):
                    select_example(image_path)
                    st.rerun()

            else:
                st.caption(f"Image not found: {image_path}")

# RIGHT SIDE: Image Input
with right_col:

    st.subheader("Upload Your Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )

    # Uploaded image takes priority over selected example
    if uploaded_file is not None:

        input_image = Image.open(
            uploaded_file
        ).convert("RGB")

    else:
        input_image = st.session_state.selected_image

    if input_image is not None:

        st.image(
            input_image,
            caption="Image for Prediction",
            use_container_width=True
        )

        if st.button(
            "Predict",
            type="primary",
            use_container_width=True
        ):

            st.session_state.prediction = process_image(
                input_image
            )

            st.rerun()

    else:
        st.info(
            "Upload an image or select an example image."
        )

    st.divider()

    if st.button(
        "Clear Selection",
        use_container_width=True
    ):
        st.session_state.selected_image = None
        st.session_state.prediction = None
        st.rerun()
