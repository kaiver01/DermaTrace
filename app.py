import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from io import BytesIO

# ---------------------------
# Page config + styling
# ---------------------------
st.set_page_config(page_title="Classification type Eczema Demo", layout="centered")

st.markdown(
    """
    <style>
      /* Big uploader frame */
      [data-testid="stFileUploader"] section {
        padding: 26px 18px !important;
        border-radius: 16px !important;
        border: 2px dashed rgba(255,255,255,0.22) !important;
        background: rgba(255,255,255,0.04) !important;
      }
      [data-testid="stFileUploader"] section:hover {
        border-color: rgba(255,255,255,0.35) !important;
        background: rgba(255,255,255,0.06) !important;
      }
      [data-testid="stFileUploader"] label {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
      }
      .tipbox {
        padding: 14px 16px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(70, 120, 255, 0.12);
        margin-top: 10px;
      }
      .muted {
        opacity: 0.75;
        font-size: 0.95rem;
        line-height: 1.45;
      }
      .section-space { margin-top: 6px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🩺 Eczema vs Atopic Dermatitis (AD) — Demo")

# ---------------------------
# Settings
# ---------------------------
MODEL_PATH = "eczema_classifier_final.keras"
IMG_SIZE = (224, 224)
LABEL = {0: "AD", 1: "Eczema"}
THRESH = 0.40

BACKBONE_NAME = "mobilenetv2_1.00_224"
CAM_LAYER_NAME = "Conv_1"

# ---------------------------
# Session state (pages)
# ---------------------------
if "page" not in st.session_state:
    st.session_state.page = "upload"   # upload -> preview -> results
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "show_cam" not in st.session_state:
    st.session_state.show_cam = True


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_pil(img: Image.Image):
    img_resized = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img_resized).astype(np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr, img_resized


def severity_proxy(p: float, pred: int) -> str:
    if pred == 0:
        return "None"
    if p < 0.65:
        return "Light"
    elif p < 0.80:
        return "Mild"
    return "Severe"


# ---------------------------
# Grad-CAM (robust / connected graph)
# ---------------------------
def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model) -> np.ndarray:
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    backbone = model.get_layer(BACKBONE_NAME)

    gap_layer = next(l for l in model.layers if isinstance(l, tf.keras.layers.GlobalAveragePooling2D))
    dropout_layer = next(l for l in model.layers if isinstance(l, tf.keras.layers.Dropout))
    dense_layer = next(l for l in model.layers if isinstance(l, tf.keras.layers.Dense))

    backbone_multi = tf.keras.Model(
        inputs=backbone.input,
        outputs=[backbone.get_layer(CAM_LAYER_NAME).output, backbone.output],
        name="backbone_multi"
    )

    with tf.GradientTape() as tape:
        conv_acts, backbone_out = backbone_multi(img_tensor, training=False)

        x = gap_layer(backbone_out)
        x = dropout_layer(x, training=False)
        preds = dense_layer(x)
        preds = tf.convert_to_tensor(preds)

        class_channel = preds[:, 0]  # prob(Eczema)

    grads = tape.gradient(class_channel, conv_acts)
    if grads is None:
        raise RuntimeError("Gradients are None. (Unexpected for this architecture.)")

    pooled_grads = tf.reduce_mean(grads[0], axis=(0, 1))
    conv_acts = conv_acts[0]

    heatmap = tf.reduce_sum(conv_acts * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def overlay_heatmap_on_image(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.35) -> Image.Image:
    img = np.array(pil_img.convert("RGB"))
    h, w, _ = img.shape

    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (1 - alpha) * img + alpha * heatmap_color
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


# ---------------------------
# Load model
# ---------------------------
try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model from '{MODEL_PATH}'.\n\nError: {e}")
    st.stop()


# ---------------------------
# PAGE 1: Upload
# ---------------------------
if st.session_state.page == "upload":
    uploaded = st.file_uploader(
        "Upload a skin image (JPG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

    st.markdown(
        '<div class="tipbox">💡 <b>Tip:</b> For best results, upload a close-up crop where the affected skin fills most of the image.</div>',
        unsafe_allow_html=True
    )

    if uploaded is not None:
        st.session_state.img_bytes = uploaded.getvalue()
        st.session_state.page = "preview"
        st.rerun()


# ---------------------------
# PAGE 2: Preview
# ---------------------------
elif st.session_state.page == "preview":
    img = Image.open(BytesIO(st.session_state.img_bytes))

    st.subheader("Preview")
    st.image(img, caption="Uploaded image preview", use_container_width=True)

    st.session_state.show_cam = st.checkbox("Show Grad-CAM (explainability)", value=st.session_state.show_cam)

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("⬅️ Choose another image"):
            st.session_state.page = "upload"
            st.session_state.img_bytes = None
            st.rerun()

    with c2:
        if st.button("🔎 Inspect image", type="primary"):
            st.session_state.page = "results"
            st.rerun()


# ---------------------------
# PAGE 3: Results
# ---------------------------
elif st.session_state.page == "results":
    img = Image.open(BytesIO(st.session_state.img_bytes))

    x, img_resized = preprocess_pil(img)

    p = float(model.predict(x, verbose=0)[0][0])
    pred = 1 if p >= THRESH else 0
    sev = severity_proxy(p, pred)
    pred_name = LABEL[pred]

    st.subheader("Results")

    resL, resR = st.columns([1.05, 1.0])
    with resL:
        st.image(img, caption="Uploaded image", use_container_width=True)

    with resR:
        top1, top2 = st.columns(2)
        with top1:
            st.metric("Probability (Eczema)", f"{p:.3f}")
        with top2:
            st.metric("Predicted label", pred_name)

        st.metric("Severity (proxy)", sev)
        st.markdown('<div class="muted">Severity is a demo proxy label (not a clinical severity score).</div>',
                    unsafe_allow_html=True)

    if st.session_state.show_cam:
        st.markdown('<div class="section-space"></div>', unsafe_allow_html=True)
        st.subheader("Grad-CAM (Explainability)")

        try:
            heatmap = make_gradcam_heatmap(x, model)
            cam_img = overlay_heatmap_on_image(img_resized, heatmap, alpha=0.35)

            camL, camR = st.columns([1.05, 1.0])
            with camL:
                st.image(cam_img, caption="Grad-CAM overlay", use_container_width=True)

            with camR:
                st.markdown(
                    """
                    **How to read this heatmap**
                    - **Red / yellow** areas contributed *most* to the model’s decision.
                    - **Blue / purple** areas contributed *least*.
                    - Use this to check if the model focuses on **skin regions** instead of background cues.
                    """.strip()
                )
                st.markdown(
                    f'<div class="muted">Layer used: <b>{BACKBONE_NAME} · {CAM_LAYER_NAME}</b>. This is an explainability aid, not medical evidence.</div>',
                    unsafe_allow_html=True
                )

        except Exception as e:
            st.warning(f"Could not generate Grad-CAM: {e}")

    st.divider()
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("⬅️ Back to preview"):
            st.session_state.page = "preview"
            st.rerun()
    with c2:
        if st.button("🔁 Upload another image", type="secondary"):
            st.session_state.page = "upload"
            st.session_state.img_bytes = None
            st.rerun()


st.caption("Note: This demo is for educational purposes and does not provide medical diagnosis.")











