# from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests

# tool create
@tool
def multiply(a: int, b: int) -> int:
  """Given 2 numbers a and b this tool returns their product"""
  return a * b


# llm = ChatOpenAI()

##<<<<<<<<<<<<<<<<<<<<<--------------- THIS IS THE TOOL BINDING --------------------->>>>>>>>>>>>>>>>>>>>>

# llm_with_tools = llm.bind_tools([multiply])

# -------- THIS IS TOOL CALLING --------->>>
# result = llm_with_tools.invoke('can you multiply 3 with 1000')


# query = HumanMessage('can you multiply 3 with 1000')
# messages = [query]
# messages

# result = llm_with_tools.invoke(messages)
# messages.append(result)
# messages

# tool_result = multiply.invoke(result.tool_calls[0])
# tool_result
# messages.append(tool_result)
# messages

# llm_with_tools.invoke(messages).content


