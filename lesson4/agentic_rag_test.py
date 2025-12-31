from langchain_tavily import TavilySearch

web_search = TavilySearch(max_results = 2)
search_result = web_search.invoke("介绍一下 LangChain 这个框架")

print(search_result)