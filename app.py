import streamlit as st
from transformers import pipeline
st.markdown("""
    <style>
   
    /* የባተኑ (Button) ከለር */
    div.stButton > button:first-child {
        background-color: #1656AD;
        color: white;
        border-radius: 5px;
        height: 3em;
        width: 100%;
    }
    /* ባተኑ ላይ ማውዝ ሲያርፍ (Hover) */
    div.stButton > button:first-child:hover {
        background-color: #808080;
        color: white;
        border: 2px solid white;
    }
    </style>
    """, unsafe_allow_html=True)
# ሞዴሉን መጫን
@st.cache_resource

def load_model():
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    return pipeline("sentiment-analysis", model=model_name)

classifier = load_model()

# አሁን AIው የሚሰጣቸውን ቃላት (NEUTRAL, POSITIVE, NEGATIVE) በትክክል አክለናል
labels_map = {
    "NEUTRAL": "NEUTRAL 😐",
    "POSITIVE": "POSITIVE 😊",
    "NEGATIVE": "NEGATIVE 😞",
    "LABEL_0": "NEGATIVE 😞",
    "LABEL_1": "NEUTRAL 😐",
    "LABEL_2": "POSITIVE 😊"
}

st.set_page_config(page_title="Smart AI Analyzer", layout="centered")

st.title("Smart Sentiment Analysis App 🤖")
st.write("Enter text (English or Amharic):")

user_input = st.text_area("Input:", placeholder="Type here...")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter text!")
    else:
        with st.spinner('AIው እያሰበ ነው...'):
            prediction = classifier(user_input)
            # የ AIውን ውጤት ወደ ትልቅ ፊደል እንቀይራለን (ለምሳሌ neutral -> NEUTRAL)
            label_id = str(prediction[0]['label']).upper() 
            score = prediction[0]['score']
            
            # ስሙን ከ map ውስጥ መፈለግ
            final_result = labels_map.get(label_id, label_id) # ካልተገኘ ራሱን label_idን ያሳያል
            
            # ውጤቱን በሚያምር ከለር ማሳየት
            if "POSITIVE" in final_result:
                st.success(f"Result: {final_result}")
            elif "NEGATIVE" in final_result:
                st.error(f"Result: {final_result}")
            else:
                st.warning(f"Result: {final_result}")
                
            st.write(f"Confidence Level: {round(score * 100, 2)}%")