# addding this line to test

# import warnings
# warnings.filterwarnings("ignore")

# from transformers import logging
# logging.set_verbosity_error()

# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import requests

search_tool = DuckDuckGoSearchRun()

result = search_tool.invoke('top news in india today')

print(result)

# Smollm = HuggingFacePipeline.from_model_id(    
#     model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
#     task="text-generation",
#     pipeline_kwargs={
#         "temperature": 0.2,
#         "return_full_text": False
#     }
# )

# model = ChatHuggingFace(llm=Smollm)