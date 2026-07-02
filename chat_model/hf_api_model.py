from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
 
load_dotenv()
 
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    task = "text-generation",
    max_new_tokens=20,
    temperature=0.2
)
  
model = ChatHuggingFace(llm=llm) 
 
result = model.invoke("what is the capital of india")
 
print(result.content)