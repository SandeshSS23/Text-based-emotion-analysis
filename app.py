import streamlit as st

import pandas as pd
import numpy as np
import altair as alt

import joblib

pipe_lr = joblib.load(open("model/best_algorithm_test2.pkl", "rb"))

emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔","sadness": "😔", "shame": "😳", "surprise": "😮"}

opposite_emotions = {
    "anger": "joy",
    "disgust": "happy",
    "fear": "surprise",
    "joy": "sadness",
    "sadness": "joy",
    "surprise": "fear"
}

def predict_emotions(docx):
    # Check for negation words
    negation_words = ["not", "no", "never", "none", "neither", "nor", "nobody", "nowhere", "nothing"]
    negation_flag = False
    for word in docx.split():
        if word.lower() in negation_words:
            negation_flag = True
            break

    results = pipe_lr.predict([docx])
    
    if negation_flag:
        results = [opposite_emotions[emotion] for emotion in results]
    
    return results, negation_flag


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions In Text")
    col1, col2 = st.columns(2)
    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        # Predict emotions
        prediction, negation_flag = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)
            st.success("Prediction")

            for emo in prediction:
                emoji_icon = emotions_emoji_dict.get(emo, "❓")
            st.write("{}: {}".format(emo, emoji_icon))
            st.write("Confidence: {}".format(np.max(probability)))

        # with col2:
        #     st.success("Prediction Probability")
        #     proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)

        #     if negation_flag:
        #         opposite_predictions = [opposite_emotions[emotion] for emotion in prediction]
        #         proba_df.index = opposite_predictions

        #     proba_df_clean = proba_df.T.reset_index()
        #     proba_df_clean.columns = ["emotions", "probability"]

        #     fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
        #     st.altair_chart(fig, use_container_width=True)




if __name__ == '__main__':
    main()