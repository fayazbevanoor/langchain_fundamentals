import warnings
warnings.filterwarnings("ignore")

from transformers import logging
logging.set_verbosity_error()

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

Smollm = HuggingFacePipeline.from_model_id(
    
    model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.2,
        "return_full_text": False
    }
)

model = ChatHuggingFace(llm=Smollm)

parser = JsonOutputParser()

template = PromptTemplate(
    template ='Give me the name, age, gender and city of a fictional person \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({})

print(result)


# ----------------  THIS IS WITHOUT CHAIN   -------------------

# #invoking prompt and filling the format instruction
# prompt = template.invoke({})

# result = model.invoke(prompt)

# #parsing the result to json
# final = parser.parse(result.content)

# print(final)