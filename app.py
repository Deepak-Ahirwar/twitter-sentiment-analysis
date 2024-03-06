from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext

def analyze(x):
    if x > 0:
        return 'Positive'
    elif x < 0:
        return 'Negative'
    else:
        return 'Neutral'

st.header('Sentiment Analysis')

def display_sentiment(sentiment_label):
    if sentiment_label == 'Positive':
        emoji = '😃'
        color = 'green'
    elif sentiment_label == 'Negative':
        emoji = '😞'
        color = 'red'
    else:
        emoji = '😐'
        color = 'yellow'

    st.write(f'<p style="color:{color}; font-size:20px;">Sentiment: {sentiment_label} {emoji}</p>', unsafe_allow_html=True)

with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        blob = TextBlob(text)
        sentiment_label = analyze(blob.sentiment.polarity)
        display_sentiment(sentiment_label)
        st.write('Subjectivity: ', round(blob.sentiment.subjectivity, 2))

    pre = st.text_input('Clean Text: ')
    if pre:
        st.write(cleantext.clean(pre, clean_all=False, extra_spaces=True,
                                 stopwords=True, lowercase=True, numbers=True, punct=True))

with st.expander('Analyze CSV'):
    upl = st.file_uploader('Upload file')

    def score(x):
        blob1 = TextBlob(x)
        return blob1.sentiment.polarity

    def analyze(x):
        if x >= 0.5:
            return 'Positive'
        elif x <= -0.5:
            return 'Negative'
        else:
            return 'Neutral'

    if upl:
        # Button to trigger sentiment analysis
        if st.button("Perform Sentiment Analysis"):
            df = pd.read_excel(upl)
            del df['Unnamed: 0']
            df['score'] = df['tweets'].apply(score)
            df['analysis'] = df['score'].apply(analyze)

            # Apply styles to the 'analysis' column
            df['analysis'] = df['analysis'].apply(style_sentiment)

            st.write(df.head(10), unsafe_allow_html=True)

            @st.cache
            def convert_df(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df)

            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='sentiment.csv',
                mime='text/csv',
            )
