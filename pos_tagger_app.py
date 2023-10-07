import streamlit as st
from main import PosHMM  # Assuming your PosHMM class is in a file named pos_hmm.py

# Initialize the POS tagger model
pos = PosHMM(debug=False)
cm, possible_tags = pos.train()  # Train the model (you can optionally skip this if already trained)

# Streamlit UI
st.title("POS Tagger Application")

# Input text area for user to enter sentences
user_input = st.text_area("Enter a sentence:")

if user_input:
    # Process the input sentence and predict POS tags
    pos_tags = pos.predict(user_input)

    # Display the user input sentence
    st.write("User Input Sentence:")
    st.markdown(f"> {user_input}")

    # Display the POS-tagged sentence with white background and black text
    tagged_sentence = ""
    for word, tag in zip(user_input.split(), pos_tags):
        tagged_sentence += f'<span style="background-color: white; padding: 4px; border-radius: 4px; color: black;">{word} ({tag})</span> '
    st.markdown(tagged_sentence, unsafe_allow_html=True)

