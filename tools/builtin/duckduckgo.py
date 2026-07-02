from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

result = search_tool.invoke("ethanol blend in india")

print(result)

# from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# search = DuckDuckGoSearchAPIWrapper()

# result = search.run("ethanol blend in india")
# print(result)