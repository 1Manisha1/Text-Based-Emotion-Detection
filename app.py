import streamlit as st
import altair as alt
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from track_utils import create_page_visited_table, add_page_visited_details, view_all_page_visited_details, add_prediction_details, view_all_prediction_details, create_emotionclf_table, IST

# Custom CSS for Dark Theme
st.markdown(
    """
    <style>
    /* Set a darker background */
    .main {
        background-color: #2b2b2b;
        color: white;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1c1c1c;
        color: white;
    }
    /* Improve button and text area appearance */
    button {
        border-radius: 8px;
        padding: 10px;
    }
    textarea {
        background-color: #333333;
        color: white;
        border-radius: 8px;
        border: none;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #F39C12;
    }
    .stButton>button {
        background-color: #F39C12;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load Model
pipe_lr = joblib.load(open("./models/emotion_classifier_pipe_lr.pkl", "rb"))

# Function
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"}

# Main Application
def main():
    st.title("🎭 Emotion Classifier App")
    menu = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("📂 Menu", menu)
    create_page_visited_table()
    create_emotionclf_table()
    if choice == "Home":
        add_page_visited_details("Home", datetime.now(IST))
        st.subheader("🔍 Emotion Detection in Text")

        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("✏️ Type Your Text Here")
            submit_text = st.form_submit_button(label='🚀 Submit')

        if submit_text:
            col1, col2 = st.columns(2)

            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            add_prediction_details(raw_text, prediction, np.max(probability), datetime.now(IST))

            with col1:
                st.success("📜 Original Text")
                st.write(raw_text)

                st.success("🎯 Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write(f"{prediction}: {emoji_icon}")
                st.write(f"Confidence: {np.max(probability):.2f}")

            with col2:
                st.success("📊 Prediction Probability")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["Emotions", "Probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(
                    x=alt.X('Emotions', sort='-y'), 
                    y='Probability', 
                    color='Emotions'
                ).properties(width=300, height=300)
                st.altair_chart(fig, use_container_width=True)

    elif choice == "Monitor":
        add_page_visited_details("Monitor", datetime.now(IST))
        st.subheader("📊 Monitor App")

        with st.expander("📄 Page Metrics"):
            page_visited_details = pd.DataFrame(view_all_page_visited_details(), columns=['Page Name', 'Time of Visit'])
            st.dataframe(page_visited_details)

            pg_count = page_visited_details['Page Name'].value_counts().rename_axis('Page Name').reset_index(name='Counts')
            c = alt.Chart(pg_count).mark_bar().encode(
                x='Page Name', 
                y='Counts', 
                color=alt.Color('Page Name', legend=None)
            )
            st.altair_chart(c, use_container_width=True)

            p = px.pie(pg_count, values='Counts', names='Page Name', title="Page Visit Distribution")
            st.plotly_chart(p, use_container_width=True)

        with st.expander('📄 Emotion Classifier Metrics'):
            df_emotions = pd.DataFrame(view_all_prediction_details(), columns=['Rawtext', 'Prediction', 'Probability', 'Time_of_Visit'])
            st.dataframe(df_emotions)

            prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
            pc = alt.Chart(prediction_count).mark_bar().encode(
                x='Prediction', 
                y='Counts', 
                color=alt.Color('Prediction', legend=None)
            )
            st.altair_chart(pc, use_container_width=True)

    else:
        add_page_visited_details("About", datetime.now(IST))

        st.write("🎉 Welcome to the Emotion Detection in Text App!")
        st.write("Analyze and visualize emotional content hidden within text with an enhanced interface.")

if __name__ == '__main__':
    main()
