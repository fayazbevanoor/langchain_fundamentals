import warnings
warnings.filterwarnings("ignore")

from transformers import logging
logging.set_verbosity_error()

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel
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
    template="genrate a tweet about {topic}",
    input_variables=['topic']
)
model = ChatHuggingFace(llm=Smollm)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template="genrate a slogan about {topic}",
    input_variables=['topic']
)


parallel_chain = RunnableParallel({
    'tweet':RunnableSequence(prompt1,model,parser),
    'slogan':RunnableSequence(prompt2,model,parser)
})

result = parallel_chain.invoke({'topic':'car'})

print('Tweet= ', result['tweet'])
print('slogan= ',result['slogan'])