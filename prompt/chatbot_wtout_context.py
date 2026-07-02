# THIS LLLM CODE IS WITHOUT THE CONEXT (CHAT HISTORY EACH CONVO IS INDEPENDENT NO CONTEXT AWARE TEXT GENERATION)
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


def load_model():
    Smollm = HuggingFacePipeline.from_model_id(
        model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        task="text-generation",
        pipeline_kwargs={
            "temperature":0.2,
            "max_new_tokens":20,
            "return_full_text": False

        }
    )
    return ChatHuggingFace(llm=Smollm)

model = load_model()

while True:
    user_input = input("you : ")
    if user_input == "exit":
        break
    else:
        result = model.invoke(user_input)
        print("AI : ",result.content)
