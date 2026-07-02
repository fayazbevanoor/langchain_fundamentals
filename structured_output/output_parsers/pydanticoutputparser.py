# The problem is SmolLM2-135M is too small to follow the formatting instructions.so output is not expected till model can run the chain

import warnings
warnings.filterwarnings("ignore")

from transformers import logging
logging.set_verbosity_error()

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

Smollm = HuggingFacePipeline.from_model_id(
    
    model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.2,
        "return_full_text": False
    }
)

model = ChatHuggingFace(llm=Smollm)

class Person(BaseModel):
    name: str = Field(description="name of the person")
    age: int = Field(gt=18,description="age of the person")
    city: str = Field(description="name of he city the person belongs to")


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template ='Give me the name, age and  city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)


# chain = template | model
# result = chain.invoke({'place':'mumbai'})
# print(result.content)
chain = template | model | parser

result = chain.invoke({'place':'mumbai'})

print(result)
