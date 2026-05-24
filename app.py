import streamlit as st
import streamlit.components.v1 as components

from transformers import pipeline

st.set_page_config(
    page_title="Biomedical NER Research System",
    page_icon="🧬",
    layout="centered"
)

st.markdown("""
<style>

.stApp{
    background:linear-gradient(to bottom,#f4f9ff,#e8f1ff);
}

.main-title{
    color:black;
    text-align:center;
    font-size:42px;
    font-weight:800;
    margin-bottom:10px;
}

.sub-text{
    color:black;
    text-align:center;
    font-size:18px;
    margin-bottom:30px;
}

label{
    color:black !important;
    font-weight:bold;
}

.stTextArea textarea{
    background-color:white;
    color:black;
    border-radius:12px;
    border:2px solid #90caf9;
    padding:12px;
    font-size:16px;
}

.stButton button{
    background-color:#1976d2;
    color:white;
    border-radius:10px;
    border:none;
    padding:10px 20px;
    font-size:16px;
    font-weight:bold;
    width:100%;
}

.stButton button:hover{
    background-color:#1565c0;
    color:white;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-title">
🧬 Biomedical NER Research System
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sub-text">
Clinical NLP Disease Entity Recognition using Medical Transformers
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():

    ner_pipeline = pipeline(
        "ner",
        model="Clinical-AI-Apollo/Medical-NER",
        aggregation_strategy="simple"
    )

    return ner_pipeline

ner = load_model()

text = st.text_area(
    "Enter Clinical Text",
    "covid and diabetes are dangerous diseases and fever is common"
)

if st.button("Analyze Clinical Text"):

    results = ner(text)

    detected_entities = []

    for item in results:

        entity_word = item["word"]

        entity_word = entity_word.replace("##", "")

        entity_word = entity_word.lower()

        detected_entities.append(entity_word)

    ignore_words = [
        "and",
        "or",
        "the",
        "is",
        "are",
        "was",
        "with",
        "from",
        "patient",
        "common",
        "dangerous"
    ]

    detected_entities = [
        word for word in detected_entities
        if word not in ignore_words
    ]

    words = text.split()

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

    for word in words:

        clean_word = word.lower()

        clean_word = clean_word.replace(",", "")

        clean_word = clean_word.replace(".", "")

        if clean_word in detected_entities:

            result_html += f"""
            <div class="disease-box">

                🔴

                <div class="disease-token">
                {word}
                </div>

                <div class="label">
                Disease Entity
                </div>

            </div>
            """

        else:

            result_html += f"""
            <div class="normal-box">

                🔵

                <div class="normal-token">
                {word}
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