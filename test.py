import warnings
warnings.filterwarnings("ignore")

from transformers import logging
logging.set_verbosity_error()

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

Smollm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.5,
        "max_new_tokens": 20,
        "return_full_text": False
    }
)

model = ChatHuggingFace(llm=Smollm)

result = model.invoke("What is Generative AI?")

print("\n===== RESPONSE =====\n")
print(result.content)