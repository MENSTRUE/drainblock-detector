from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT
    / "models"
    / "drainblock_mobilenetv3_FULL_v28.pth"
)

ASSET_DIR = ROOT / "assets"
ASSET_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_DIR = ASSET_DIR / "results"
RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MODEL
# ============================================================

def build_model(
    num_classes: int,
) -> nn.Module:

    model = mobilenet_v3_small(
        weights=None
    )

    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features,
        num_classes,
    )

    return model


@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    class_names = checkpoint.get(
        "class_names",
        [
            "clean_context",
            "debris",
        ],
    )

    image_size = int(
        checkpoint.get(
            "img_size",
            224,
        )
    )

    model = build_model(
        len(class_names)
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(
        DEVICE
    )

    model.eval()

    metadata = {
        "class_names":
            class_names,

        "image_size":
            image_size,

        "best_validation_f1":
            checkpoint.get(
                "best_validation_f1"
            ),

        "test_metrics":
            checkpoint.get(
                "test_metrics",
                {},
            ),
    }

    return (
        model,
        metadata,
    )


# ============================================================
# PREPROCESS
# ============================================================

def build_transform(
    image_size: int,
):

    return transforms.Compose(
        [
            transforms.Resize(
                (
                    image_size,
                    image_size,
                )
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                [
                    0.485,
                    0.456,
                    0.406,
                ],
                [
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )


# ============================================================
# INFERENCE
# ============================================================

@torch.inference_mode()
def predict(
    image: Image.Image,
    model: nn.Module,
    metadata: dict,
):

    transform = build_transform(
        metadata[
            "image_size"
        ]
    )

    tensor = (
        transform(
            image.convert("RGB")
        )
        .unsqueeze(0)
        .to(DEVICE)
    )

    logits = model(
        tensor
    )

    probs = torch.softmax(
        logits,
        dim=1,
    )[0]

    pred_idx = int(
        probs.argmax()
        .item()
    )

    class_names = metadata[
        "class_names"
    ]

    prediction = (
        class_names[
            pred_idx
        ]
    )

    confidence = float(
        probs[
            pred_idx
        ].item()
    )

    probabilities = {
        class_names[i]:
            float(
                probs[i]
                .item()
            )
        for i in range(
            len(class_names)
        )
    }

    return {
        "prediction":
            prediction,

        "confidence":
            confidence,

        "probabilities":
            probabilities,
    }


# ============================================================
# RESULT IMAGE
# ============================================================

def create_result_image(
    image: Image.Image,
    result: dict,
):

    image = image.convert(
        "RGB"
    )

    canvas = image.copy()

    draw = ImageDraw.Draw(
        canvas
    )

    prediction = result[
        "prediction"
    ]

    confidence = (
        result[
            "confidence"
        ]
        * 100
    )

    if prediction == "debris":
        status = (
            "Potential debris detected"
        )
    else:
        status = (
            "No dominant debris signal"
        )

    text = (
        f"{prediction} | "
        f"{confidence:.2f}%\n"
        f"{status}"
    )

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            28,
        )
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.multiline_textbbox(
        (
            20,
            20,
        ),
        text,
        font=font,
        spacing=6,
    )

    padding = 12

    box = (
        bbox[0] - padding,
        bbox[1] - padding,
        bbox[2] + padding,
        bbox[3] + padding,
    )

    draw.rectangle(
        box,
        fill="black",
    )

    draw.multiline_text(
        (
            20,
            20,
        ),
        text,
        fill="white",
        font=font,
        spacing=6,
    )

    return canvas


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(
    image: Image.Image,
    result: dict,
):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    image_path = (
        RESULT_DIR
        / f"result_{timestamp}.jpg"
    )

    json_path = (
        RESULT_DIR
        / f"result_{timestamp}.json"
    )

    image.save(
        image_path,
        quality=95,
    )

    json_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    return (
        image_path,
        json_path,
    )


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="DrainBlock Detector",
    page_icon="🗑️",
    layout="centered",
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model, metadata = load_model()

except Exception as exc:

    st.error(
        f"Gagal memuat model:\n\n{exc}"
    )

    st.stop()


# ============================================================
# UI
# ============================================================

st.title(
    "DrainBlock Detector"
)

st.caption(
    "Weakly supervised debris-screening prototype"
)

st.write(
    """
Upload gambar area jalan, selokan, atau lingkungan luar ruang.
Model akan memperkirakan apakah gambar lebih dekat ke
**debris** atau **clean context**.
"""
)


uploaded_file = st.file_uploader(
    "Upload gambar",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert(
        "RGB"
    )

    st.subheader(
        "Input"
    )

    st.image(
        image,
        use_container_width=True,
    )

    if st.button(
        "Analyze",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "Analyzing image..."
        ):

            result = predict(
                image=image,
                model=model,
                metadata=metadata,
            )

            result_image = create_result_image(
                image=image,
                result=result,
            )

            image_path, json_path = save_result(
                image=result_image,
                result=result,
            )

        st.success(
            "Analysis complete."
        )

        st.subheader(
            "Result"
        )

        st.image(
            result_image,
            use_container_width=True,
        )

        col1, col2 = st.columns(
            2
        )

        with col1:

            st.metric(
                "Prediction",
                result[
                    "prediction"
                ],
            )

        with col2:

            st.metric(
                "Confidence",
                f"{result['confidence'] * 100:.2f}%",
            )

        st.subheader(
            "Class probabilities"
        )

        for class_name, probability in (
            result[
                "probabilities"
            ].items()
        ):

            st.write(
                f"**{class_name}**"
            )

            st.progress(
                float(
                    probability
                )
            )

            st.caption(
                f"{probability * 100:.2f}%"
            )

        st.info(
            "Model confidence bukan persentase penyumbatan fisik "
            "dan bukan ukuran severity drainase."
        )

        st.caption(
            f"Saved image: {image_path}"
        )

        st.caption(
            f"Saved JSON: {json_path}"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Model Info"
    )

    st.write(
        f"**Device:** {DEVICE}"
    )

    st.write(
        f"**Input size:** "
        f"{metadata['image_size']}×"
        f"{metadata['image_size']}"
    )

    st.write(
        "**Classes:**"
    )

    for class_name in metadata[
        "class_names"
    ]:

        st.write(
            f"- {class_name}"
        )

    st.divider()

    st.caption(
        "DrainBlock 6% — MobileNetV3-Small"
    )