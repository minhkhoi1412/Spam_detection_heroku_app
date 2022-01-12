import streamlit as st
import pandas as pd
import numpy as np
from model_building import entity_tagging, utils
from datetime import datetime
import pickle


st.write("""
# Spam Detection App
This app predicts spam messages!
""")

st.sidebar.header('User Input Messages')

st.sidebar.markdown("""
[Example input](https://fptcloud.sharepoint.com/:x:/s/SCC-DA-Team/ETE-mLqbMgpFvwbD4J0FxNEBoiV0BMV5kJA1aPHCjiumOw?e=UWS4ip)
""")

# Reads in saved classification model
load_clf = pickle.load(open('./bow_nb_clf.pkl', 'rb'))

input_message = st.text_input('Input your message here:')
uploaded_file = st.sidebar.file_uploader("Upload your input Excel file", type=["xlsx"])

if uploaded_file is not None:
    input_df = pd.read_excel(uploaded_file, dtype={'msg': str})
    messages_raw = pd.read_excel('./message_modified_v1.2.xlsx', dtype={'msg': str})
    messages = messages_raw.drop(columns='class')
    df = pd.concat([input_df, messages], ignore_index=True)
    list_content = df['msg'].to_list()
    list_content = entity_tagging.entity_tagging(list_content)
    list_content_vec, list_len_sms, dictionary = utils.vectorize(list_content, 'bow')
    label_predict = load_clf.predict(list_content_vec[:input_df.shape[0]])
    label_predict = label_predict.tolist()

    for i in range(len(label_predict)):
        if label_predict[i] == 0:
            label_predict[i] = 'ham'

        if label_predict[i] == 1:
            label_predict[i] = 'spam'

    df = pd.DataFrame(data={'Message': input_df['msg'].to_list(), 'Label': label_predict})
    st.write(df)

elif len(input_message) > 0:
    messages_raw = pd.read_excel('./message_modified_v1.2.xlsx', dtype={'msg': str})
    list_content = messages_raw['msg'].to_list()
    list_content.insert(0, input_message)
    list_content = entity_tagging.entity_tagging(list_content)
    list_content_vec, list_len_sms, dictionary = utils.vectorize(list_content, 'bow')
    label_predict = load_clf.predict(list_content_vec[:1])

    if label_predict[0] == 0:
        label = 'ham'
    else:
        label = 'spam'

    st.write('This is %s message' %(label))

