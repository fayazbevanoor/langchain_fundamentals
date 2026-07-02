import warnings
warnings.filterwarnings("ignore")

from transformers import logging
logging.set_verbosity_error()

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

Smollm = HuggingFacePipeline.from_model_id(
    
    model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.2,
        "return_full_text": False
    }
)

model = ChatHuggingFace(llm=Smollm)

#prompt-> detailed report 
template = PromptTemplate(
    template='write 2 line brief on {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

chain = template | model | parser

result =chain.invoke({'topic':'gen-ai'})

print(result)