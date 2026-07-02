import streamlit as st
from langchain_community.document_loaders import PyPDFLoader


st.title("Resume Reader")

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type="pdf"
)

if uploaded_file:
    st.success("PDF Uploaded Successfully")
    print("File Name:", uploaded_file.name)
    print("File Type:", uploaded_file.type)
    print("File Size:", uploaded_file.size)

    print(type(uploaded_file))

    with open("temp_resume.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    print("PDF saved successfully")

#-------------- Document loader creation --------------
loader = PyPDFLoader("temp_resume.pdf")
docs = loader.load()

print(type(docs))
print(len(docs))