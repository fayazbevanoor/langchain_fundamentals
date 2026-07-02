import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

st.set_page_config(
    page_title="My AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 My AI Assistant")
st.markdown("Ask me anything!")

@st.cache_resource
def load_model():
    Smollm = HuggingFacePipeline.from_model_id(
        model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        task="text-generation",
        pipeline_kwargs={
            "temperature": 0.5,
            "max_new_tokens": 500,
            "return_full_text": False
        }
    )
    return ChatHuggingFace(llm=Smollm)

with st.spinner("Loading AI Model..."):
    model = load_model()

user_input = st.text_input(
    "Enter your prompt",
    placeholder="e.g. What is Generative AI?"
)

if st.button("🚀 Submit", use_container_width=True):
    if user_input:
        with st.spinner("Thinking..."):
            result = model.invoke(user_input)

        st.success("Response Generated")

        st.markdown("### 🤖 AI Response")
        st.write(result.content)

    else:
        st.warning("Please enter a prompt")