import re
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from urlextract import URLExtract
from wordcloud import WordCloud
import emoji

extract = URLExtract()

def preprocess(data):
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)
    
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%Y, %H:%M - ')
    df.rename(columns={'message_date': 'date'}, inplace=True)
    
    users, messages = [], []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group_notification')
            messages.append(entry[0])
    
    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)
    
    df['only_date'] = df['date'].dt.date
    df['day_name'] = df['date'].dt.day_name()
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['period'] = df['hour'].apply(lambda h: f"{h}-{(h + 1) % 24}")
    
    return df

def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    num_messages = df.shape[0]
    words = sum(len(msg.split()) for msg in df['message'])
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]
    num_links = sum(len(extract.find_urls(msg)) for msg in df['message'])
    
    return num_messages, words, num_media_messages, num_links

def create_wordcloud(selected_user, df):
    with open('stop_hinglish.txt', 'r') as f:
        stop_words = set(f.read().split())
    
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    df = df[(df['user'] != 'group_notification') & (df['message'] != '<Media omitted>\n')]
    df['message'] = df['message'].apply(lambda msg: ' '.join([word for word in msg.lower().split() if word not in stop_words]))
    
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    return wc.generate(df['message'].str.cat(sep=' '))

def most_common_words(selected_user, df):
    with open('stop_hinglish.txt', 'r') as f:
        stop_words = set(f.read().split())
    
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    df = df[(df['user'] != 'group_notification') & (df['message'] != '<Media omitted>\n')]
    words = [word for msg in df['message'] for word in msg.lower().split() if word not in stop_words]
    return pd.DataFrame(Counter(words).most_common(20))

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    emojis = [c for msg in df['message'] for c in msg if c in emoji.EMOJI_DATA]
    return pd.DataFrame(Counter(emojis).most_common())

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    timeline['time'] = timeline['month'] + '-' + timeline['year'].astype(str)
    return timeline

def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df.groupby('only_date').count()['message'].reset_index()

def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

# Streamlit UI
st.sidebar.title("WhatsApp Chat Analyzer")
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    data = uploaded_file.getvalue().decode('utf-8')
    df = preprocess(data)
    
    user_list = sorted(df['user'].unique().tolist())
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.insert(0, 'Overall')
    selected_user = st.sidebar.selectbox('Show analysis wrt', user_list)
    
    if st.sidebar.button('Show Analysis'):
        num_messages, words, num_media_messages, num_links = fetch_stats(selected_user, df)
        st.title('Top Statistics')
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Messages", num_messages)
        col2.metric("Total Words", words)
        col3.metric("Media Shared", num_media_messages)
        col4.metric("Links Shared", num_links)
        
        st.title('Monthly Timeline')
        timeline = monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)
        
        st.title('Daily Timeline')
        daily_data = daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_data['only_date'], daily_data['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)
        
        st.title('Activity Heatmap')
        heatmap = activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        sns.heatmap(heatmap, ax=ax)
        st.pyplot(fig)
        
        st.title('Word Cloud')
        wc_img = create_wordcloud(selected_user, df)
        st.image(wc_img.to_array())
        
        st.title('Most Common Words')
        st.dataframe(most_common_words(selected_user, df))
        
        st.title('Emoji Analysis')
        emoji_df = emoji_helper(selected_user, df)
        st.dataframe(emoji_df)
