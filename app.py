import streamlit as st
from transformers import pipeline
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
import io

# --- 1. ገጹን ማስተካከል (Page Settings) ---
st.set_page_config(page_title="Smart sentiment analyis on ethiopian social media ", page_icon="🤖", layout="centered")

# --- 2. PROFESSIONAL CSS DESIGN ---
st.markdown("""
    <style>
    /* አጠቃላይ ገጽታ */
    .stApp {
        background-color: #F8F9FA;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* የርዕስ ስታይል */
    .main-title {
        color: #1656AD;
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 10px;
    }

    /* የባተኑ (Button) ዲዛይን */
    div.stButton > button:first-child {
        background-color: #1656AD;
        color: white;
        border-radius: 8px;
        height: 3.5em;
        width: 100%;
        font-weight: bold;
        border: none;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #0D3B7A;
        transform: translateY(-2px);
    }

    /* የውጤት ሳጥኖች (Custom Result Cards) */
    .result-card {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        color: white;
        margin: 10px 0;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. AI ሞዴሉን መጫን (Cache) ---
@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    return pipeline("sentiment-analysis", model=model_name)

classifier = load_model()

# የውጤት መለወጫ
labels_map = {
    "NEUTRAL": "NEUTRAL 😐", "POSITIVE": "POSITIVE 😊", "NEGATIVE": "NEGATIVE 😞",
    "LABEL_0": "NEGATIVE 😞", "LABEL_1": "NEUTRAL 😐", "LABEL_2": "POSITIVE 😊"
}

# --- 4. ፋይል የማንበቢያ ፋንክሽኖች ---
def extract_pdf_text(file):
    reader = PdfReader(file)
    return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])

def extract_docx_text(file):
    doc = Document(file)
    return " ".join([p.text for p in doc.paragraphs])

# --- 5. UI CONTENT ---
st.markdown("<h1 class='main-title'>Smart Sentiment Analysis on Ethiopian Social Media 🤖</h1>", unsafe_allow_html=True)
st.write("<p style='text-align:center;'>Multilingual Support: አማርኛ | English</p>", unsafe_allow_html=True)

# TABS መፍጠር
tab1, tab2 = st.tabs(["✍️ Single Text Analysis", "📁 File Upload Analysis"])

# --- TAB 1: አንድ በአንድ መጻፊያ ---
with tab1:
    user_input = st.text_area("Enter your comment:", placeholder="እዚህ ይጻፉ / Type here...", height=150)
    
    if st.button("Analyze Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter some text!")
        else:
            with st.spinner('AIው እያሰበ ነው...'):
                prediction = classifier(user_input[:512]) # Limit to 512 tokens
                label_id = str(prediction[0]['label']).upper()
                score = prediction[0]['score']
                final_result = labels_map.get(label_id, label_id)
                
                # Custom Color logic
                color = "#f1c40f" if "NEUTRAL" in final_result else ("#2ecc71" if "POSITIVE" in final_result else "#e74c3c")
                st.markdown(f"<div class='result-card' style='background-color:{color};'>{final_result}</div>", unsafe_allow_html=True)
                st.write(f"<p style='text-align:center;'>Confidence: {round(score * 100, 2)}%</p>", unsafe_allow_html=True)

# --- TAB 2: ፋይል መጫኛ ---
with tab2:
    st.subheader("የጅምላ ዳታ ትንተና (Sentiment Analysis)")
    uploaded_file = st.file_uploader("Upload CSV, PDF, or Word", type=["csv", "pdf", "docx"])

    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1]
        
        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
            column = st.selectbox("Select Text Column:", df.columns)
            if st.button("Run Bulk Analysis"):
                with st.spinner("AIው በሺዎች የሚቆጠሩ ዳታዎችን እያነበበ ነው..."):
                    # Process with progress bar
                    progress = st.progress(0)
                    results = []
                    for i, text in enumerate(df[column]):
                        res = classifier(str(text)[:512])[0]
                        results.append(labels_map.get(res['label'], "Unknown"))
                        progress.progress((i + 1) / len(df))
                    
                    df['Sentiment_Result'] = results
                    st.success("ተጠናቋል!")
                    st.bar_chart(df['Sentiment_Result'].value_counts())
                    st.dataframe(df)
                    st.download_button("Download CSV", df.to_csv(index=False), "ai_results.csv")

        elif file_type in ["pdf", "docx"]:
            if st.button("Analyze Document Content"):
                with st.spinner("ሰነዱን እያነበብኩ ነው..."):
                    raw_text = extract_pdf_text(uploaded_file) if file_type == "pdf" else extract_docx_text(uploaded_file)
                    st.text_area("Document Preview:", raw_text[:1000] + "...", height=100)
                    
                    # Analyze overall document sentiment (first 512 tokens)
                    res = classifier(raw_text[:512])[0]
                    final_result = labels_map.get(res['label'].upper(), res['label'])
                    
                    color = "#f1c40f" if "NEUTRAL" in final_result else ("#2ecc71" if "POSITIVE" in final_result else "#e74c3c")
                    st.markdown(f"<div class='result-card' style='background-color:{color};'>Overall Document Sentiment: {final_result}</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Developed with Group One | Powered by XLM-RoBERTa Transformer")
