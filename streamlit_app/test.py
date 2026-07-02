import streamlit as st

st.header('My AI')

user_input = st.text_input("enter you prompt")

if st.button("Submit"):
    if user_input:
        st.write("thank you")
    else:
        st.warning("Please enter a prompt")