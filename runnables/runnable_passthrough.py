import warnings
warnings.filterwarnings("ignore")

from transformers import logging
logging.set_verbosity_error()

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel, RunnablePassthrough
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
    template="which is a best indian {topic}",
    input_variables=['topic']
)
model = ChatHuggingFace(llm=Smollm)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template="genrate a explaination about this {text}",
    input_variables=['topic']
)


name_gen_chain =RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel({
    'passedthrough_car_info':RunnablePassthrough(),
    'explaination':RunnableSequence(prompt2,model,parser)
})

merged_chain = name_gen_chain | parallel_chain

result = merged_chain.invoke({'topic':'car'})

print('car_info= ', result['passedthrough_car_info'])
print('explaination= ',result['explaination'])