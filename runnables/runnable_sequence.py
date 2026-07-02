import warnings
warnings.filterwarnings("ignore")

from transformers import logging
logging.set_verbosity_error()

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from typing import Literal

Smollm = HuggingFacePipeline.from_model_id(
    
    model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.2,
        "return_full_text": False
    }
)

prompt1 = PromptTemplate(
    template="write a joke abount {topic}",
    input_variables=['topic']
)
model = ChatHuggingFace(llm=Smollm)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

chain = RunnableSequence(prompt1,model,parser,prompt2,model,parser)

#can also write as 
# chain = prompt1 | model | parser | prompt2 | model | parser

print(chain.invoke({'topic':'car'}))