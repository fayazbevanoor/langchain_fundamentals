from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
# from typing import TypedDict
 
load_dotenv()
 
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    task = "text-generation",
    max_new_tokens=30,
    temperature=0.2
)
  
model = ChatHuggingFace(llm=llm) 


class Review(BaseModel):
    summary: str = Field(description="summary of review")
    sentiment: str = Field(description="positive, negative or neutral")

parser = PydanticOutputParser(pydantic_object=Review)

prompt = f"""
Extract the following information.

Review:
The hardware is great, but the software feels bloated.

{parser.get_format_instructions()}
"""

result = model.invoke(prompt)

parsed_result = parser.parse(result.content)

print(parsed_result)