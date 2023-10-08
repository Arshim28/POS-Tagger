import streamlit as st
from main import PosHMM
import re

pos = PosHMM(debug=False)
cm, possible_tags = pos.train() 

st.title("POS Tagger Application")
user_input = st.text_area("Enter a sentence:")

if user_input:
    # Remove special characters except for necessary ones
    user_input_cleaned = re.sub(r'[^a-zA-Z0-9\s!?\'-]', '', user_input)
    
    pos_tags = pos.predict(user_input_cleaned)
    st.write("User Input Sentence:")
    st.markdown(f"> {user_input_cleaned}")
    tagged_sentence = ""
    for word, tag in zip(user_input_cleaned.split(), pos_tags):
        tagged_sentence += f'<span style="background-color: white; padding: 4px; border-radius: 4px; color: black;">{word} ({tag})</span> '
    st.markdown(tagged_sentence, unsafe_allow_html=True)
