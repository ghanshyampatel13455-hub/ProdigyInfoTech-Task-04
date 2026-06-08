import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Page Configuration
st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="✋",
    layout="wide"
)
st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#020617,#0f172a,#111827);
    color:white;
}

/* Hide Streamlit Menu */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Hero Title */
.hero-title{
    text-align:center;
    font-size:4rem;
    font-weight:800;
    line-height:1.1;
    margin-top:20px;
}

.gradient-text{
    background: linear-gradient(
    90deg,
    #60a5fa,
    #a78bfa,
    #f472b6,
    #fb7185,
    #f59e0b);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

/* Glass Cards */
.glass{
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border:1px solid rgba(255,255,255,0.1);
    border-radius:20px;
    padding:25px;
    margin-top:15px;
}

/* Stats */
.stat{
    text-align:center;
    padding:20px;
    border-radius:15px;
    background: rgba(255,255,255,0.05);
}

/* Footer */
.footer{
    text-align:center;
    padding:40px;
}

.footer img{
    margin:10px;
    transition:0.3s;
}

.footer img:hover{
    transform:scale(1.2);
}

</style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_gesture_model():
    return load_model(
    "models/hand_gesture_model.keras",
    compile=False
)

model = load_gesture_model()

# Gesture Classes
classes = {
    0: "Palm ✋",
    1: "L 🤟",
    2: "Fist ✊",
    3: "Fist Moved 👊",
    4: "Thumb 👍",
    5: "Index ☝️",
    6: "OK 👌",
    7: "Palm Moved 🖐️",
    8: "C 🤏",
    9: "Down 👇"
}

# Header
c1,c2,c3,c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="stat">
    <h1>20K+</h1>
    Images
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="stat">
    <h1>10</h1>
    Gestures
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="stat">
    <h1>100%</h1>
    Accuracy
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="stat">
    <h1>CNN</h1>
    Model
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;">

<div style="
display:inline-block;
padding:8px 18px;
border-radius:30px;
background:rgba(255,255,255,0.05);
border:1px solid rgba(255,255,255,0.1);
margin-bottom:20px;">
🚀 Prodigy InfoTech • ML Internship • Task 04
</div>

<div class="hero-title">
Hand Gesture<br>
<span class="gradient-text">Recognition</span><br>
System
</div>

<p style="font-size:20px;color:#cbd5e1;">
Deep Learning based Human Computer Interaction
using CNN & TensorFlow
</p>

</div>
""", unsafe_allow_html=True)

st.write("---")

# Sidebar
st.sidebar.header("👨‍💻 Developer Information")

st.sidebar.success("GHANSHYAM PATEL")

st.sidebar.info("B.Tech AI & DS (7th Semester)")

st.sidebar.info("Machine Learning Intern @ Prodigy InfoTech")




st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div class="glass">
<h2>📤 Upload Gesture Image</h2>
<p>Select a hand gesture image to classify using the trained CNN model.</p>
</div>
""", unsafe_allow_html=True)

# Upload Image
uploaded_file = st.file_uploader(
    "Upload a Hand Gesture Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    img = np.array(image)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img_resized = cv2.resize(img, (64, 64))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    prediction = model.predict(img_resized)

    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    with col2:

        st.success("Prediction Complete")

        st.metric(
            label="Detected Gesture",
            value=classes[class_index]
        )

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.subheader("Prediction Probabilities")

        for i, prob in enumerate(prediction[0]):
            st.progress(float(prob))
            st.write(
                f"{classes[i]} : {prob*100:.2f}%"
            )

            st.write("")
st.write("")

st.markdown("""
<h2 style='text-align:center;'>
📊 Model Performance Analytics
</h2>
""", unsafe_allow_html=True)

history_df = pd.read_csv("training_history.csv")

col1, col2 = st.columns(2)

with col1:

    fig, ax = plt.subplots(figsize=(6,4))

    fig.patch.set_facecolor("#0B1220")
    ax.set_facecolor("#0B1220")

    ax.plot(
        history_df["accuracy"],
        marker="o",
        linewidth=2
    )

    ax.plot(
        history_df["val_accuracy"],
        marker="o",
        linewidth=2
    )

    ax.set_title(
        "Model Accuracy",
        color="white"
    )

    ax.set_xlabel(
        "Epoch",
        color="white"
    )

    ax.set_ylabel(
        "Accuracy",
        color="white"
    )

    ax.tick_params(colors="white")

    ax.grid(alpha=0.2)

    st.pyplot(fig)

with col2:

    fig2, ax2 = plt.subplots(figsize=(6,4))

    fig2.patch.set_facecolor("#0B1220")
    ax2.set_facecolor("#0B1220")

    ax2.plot(
        history_df["loss"],
        marker="o",
        linewidth=2
    )

    ax2.plot(
        history_df["val_loss"],
        marker="o",
        linewidth=2
    )

    ax2.set_title(
        "Model Loss",
        color="white"
    )

    ax2.set_xlabel(
        "Epoch",
        color="white"
    )

    ax2.set_ylabel(
        "Loss",
        color="white"
    )

    ax2.tick_params(colors="white")

    ax2.grid(alpha=0.2)

    st.pyplot(fig2)

st.write("---")

st.markdown("""
<hr>

<div class="footer">

<h2>👨‍💻 Developed By</h2>

<h1 style="
background: linear-gradient(90deg,#60a5fa,#f472b6,#f59e0b);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;">
GHANSHYAM PATEL
</h1>

<p>
B.Tech Artificial Intelligence & Data Science (7th Semester)
</p>

<p>
Machine Learning Intern @ Prodigy InfoTech
</p>

<a href="https://www.linkedin.com/in/ghanshyam-patel-3a8679307" target="_blank">
<img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="50">
</a>

<a href="https://github.com/ghanshyampatel13455-hub" target="_blank">
<img src="https://cdn-icons-png.flaticon.com/512/733/733553.png" width="50">
</a>

<br><br>

<p style="color:gray;">
© 2026 GHANSHYAM PATEL • Hand Gesture Recognition System
</p>

</div>
""", unsafe_allow_html=True)
