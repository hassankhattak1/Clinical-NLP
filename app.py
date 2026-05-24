import streamlit as st
import streamlit.components.v1 as components
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification
)

st.set_page_config(
    page_title="Biomedical NER Research System",
    page_icon="🧬",
    layout="centered"
)

st.markdown("""
<style>

.stApp {
    background: linear-gradient(to bottom, #f4f9ff, #e8f1ff);
}

.main-title {
    color: black;
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 10px;
}

.sub-text {
    color: black;
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

label {
    color: black !important;
    font-weight: bold;
}

.stTextArea textarea {
    background-color: white;
    color: black;
    border-radius: 12px;
    border: 2px solid #90caf9;
    padding: 12px;
    font-size: 16px;
}

.stButton button {
    background-color: #1976d2;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-size: 16px;
    font-weight: bold;
    width: 100%;
}

.stButton button:hover {
    background-color: #1565c0;
    color: white;
}

</style>
""", unsafe_allow_html=True)

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False
)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    ignore_mismatched_sizes=True
)

model.eval()

id2label = {
    0: "O",
    1: "DISEASE"
}

st.markdown("""
<div class="main-title">
🧬 Biomedical NER Research System
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sub-text">
BioBERT-based Disease Entity Recognition using Clinical NLP
</div>
""", unsafe_allow_html=True)

text = st.text_area(
    "Enter Clinical Text",
    "the patient suffers from lung cancer and diabetes"
)

if st.button("Analyze Clinical Text"):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():

        outputs = model(**inputs)

    predictions = torch.argmax(
        outputs.logits,
        dim=-1
    )

    prediction_ids = predictions[0].detach().cpu().tolist()

    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )

    predicted_labels = []

    for p in prediction_ids:

        predicted_labels.append(
            id2label[p]
        )

    result_html = """
    <html>

    <head>

    <style>

    body{
        background:#f4f9ff;
        font-family:Arial;
    }

    .title{
        text-align:center;
        color:black;
        font-size:28px;
        font-weight:bold;
        margin-bottom:20px;
    }

    .container{
        display:flex;
        flex-wrap:wrap;
        gap:12px;
        justify-content:center;
    }

    .normal-box{
        background:white;
        padding:15px;
        border-radius:15px;
        min-width:140px;
        text-align:center;
        border-left:6px solid #1976d2;
        box-shadow:0px 2px 8px rgba(0,0,0,0.08);
    }

    .disease-box{
        background:white;
        padding:15px;
        border-radius:15px;
        min-width:140px;
        text-align:center;
        border-left:6px solid red;
        box-shadow:0px 2px 8px rgba(0,0,0,0.08);
    }

    .normal-token{
        color:black;
        font-size:18px;
        margin-top:5px;
    }

    .disease-token{
        color:red;
        font-size:20px;
        font-weight:bold;
        margin-top:5px;
    }

    .label{
        color:black;
        margin-top:8px;
    }

    </style>

    </head>

    <body>

    <div class="title">
    Prediction Results
    </div>

    <div class="container">
    """

    for token, label in zip(tokens, predicted_labels):

        if token not in ["[CLS]", "[SEP]", "[PAD]"]:

            clean_token = token.replace("##", "")

            if label == "DISEASE":

                result_html += f"""
                <div class="disease-box">

                    🔴

                    <div class="disease-token">
                    {clean_token}
                    </div>

                    <div class="label">
                    Disease
                    </div>

                </div>
                """

            else:

                result_html += f"""
                <div class="normal-box">

                    🔵

                    <div class="normal-token">
                    {clean_token}
                    </div>

                    <div class="label">
                    Normal
                    </div>

                </div>
                """

    result_html += """
    </div>

    </body>

    </html>
    """

    components.html(
        result_html,
        height=700,
        scrolling=True
    )