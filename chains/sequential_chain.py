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

#prompt1-> detailed report 
template1 = PromptTemplate(
    template='write small report on {topic}',
    input_variables=['topic']
)

#prompt2-> summary 
template2 = PromptTemplate(
    template='write 2 line summary on {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result =chain.invoke({'topic':'black hole'})

print(result)