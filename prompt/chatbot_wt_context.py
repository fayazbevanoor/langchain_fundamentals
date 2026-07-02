from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage 

def load_model():
    Smollm = HuggingFacePipeline.from_model_id(
        model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        task="text-generation",
        pipeline_kwargs={
            "temperature":0.1,
            "return_full_text": False

        }
    )
    return ChatHuggingFace(llm=Smollm)

model = load_model()

chat_history =[
    SystemMessage(content="you are a helpful AI assistant")
]

while True:
    user_input = input("you : ")
    chat_history.append(HumanMessage(content=user_input)) 
    if user_input == "exit":
        break
    else:
        result = model.invoke(chat_history)
        chat_history.append(AIMessage(content=result.content))
        print("AI : ",result.content)

print(chat_history)