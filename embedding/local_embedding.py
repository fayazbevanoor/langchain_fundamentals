from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# text = "hello world"
doc = [
    "hii",
    "how",
    "you"
]

# vector = embedding.embed_query(text)
vector = embedding.embed_documents(doc)

print(str(vector))