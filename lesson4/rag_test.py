from langchain_deepseek import ChatDeepSeek
from langchain_ollama import OllamaEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv


load_dotenv(override = True)

model = ChatDeepSeek(model = "deepseek-chat", temperature = 0)

embeddings = OllamaEmbeddings(
    model = "bge-m3"
)

# test the model and embeddings
# response = model.invoke([{"role": "user", "content": "Hello, DeepSeek!"}])
# print("Model response:", response)
# embedding_vector = embeddings.embed_query("Test embedding")
# print("Embedding vector:", embedding_vector)

# load documents
from langchain_community.document_loaders import TextLoader, Docx2txtLoader

loader = TextLoader("sample_document.txt", encoding = "utf-8")
documents = loader.load()

sensitive_loader = TextLoader("sensitive_document.txt", encoding = "utf-8")
sensitive_documents = sensitive_loader.load()

# print(documents[0].page_content)
# print(sensitive_documents[0].page_content)

# split documents
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500, # 每个块的最大字符数
    chunk_overlap = 50, # 块之间的重叠字符数
    separators = ["\n\n", "\n", " ", ""]
)

texts = text_splitter.split_documents(documents)

sensitive_texts = text_splitter.split_documents(sensitive_documents)

# print(f"after splitting, we have {len(texts)} chunks.")
# print(f"after splitting, we have {len(sensitive_texts)} sensitive chunks.")

# create vector store and query
from langchain_community.vectorstores import FAISS

vector_store = FAISS.from_documents(texts, embeddings)
vector_store.save_local("faiss_index")

vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization = True # 允许不安全的反序列化
)
print(f"save normal vector store successfully")

sensitive_vector_store = FAISS.from_documents(sensitive_texts, embeddings)
sensitive_vector_store.save_local("sensitive_faiss_index")

sensitive_vector_store = FAISS.load_local(
    "sensitive_faiss_index",
    embeddings,
    allow_dangerous_deserialization = True # 允许不安全的反序列化
)
print(f"save sensitive vector store successfully")

# load and create retriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# create bm25 retriever
bm25_retriever = BM25Retriever.from_documents(texts)
bm25_retriever.k = 3 # 设置返回的文档数量

# create ensemble retriever
faiss_retriever = vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 3} # 设置返回的文档数量
)

ensemble_retriever = EnsembleRetriever(
    retrievers = [faiss_retriever, bm25_retriever],
    weights = [0.5, 0.5] # 设置各个检索器的权重
)

print("Retrievers created successfully")

sensitive_bm25_retriever = BM25Retriever.from_documents(sensitive_texts)
sensitive_bm25_retriever.k = 3

sensitive_faiss_retriever = sensitive_vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 3}
)

sensitive_ensemble_retriever = EnsembleRetriever(
    retrievers = [sensitive_faiss_retriever, sensitive_bm25_retriever],
    weights = [0.5, 0.5]
)

print("Sensitive retrievers created successfully")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

template = """你是一个专业的问答助手。请根据以下提供的上下文信息来回答用户的问题。
如果上下文中没有相关信息，请诚实地告诉用户你不知道，不要编造答案。

上下文信息：
{context}

问题: {question}

回答:"""

prompt = ChatPromptTemplate.from_template(template)

chain = ensemble_retriever | format_docs # 定义检索链, 先检索再格式化文档

# retrieval = chain.invoke("Langchain是什么？")
# print("Retrieval result:", retrieval)
# print("=" * 60)
#
# retrieval_chain = (
#     {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt | model | StrOutputParser()
# ) # 定义完整的检索问答链, 先检索再回答
#
# content = retrieval_chain.invoke("Langchain是什么？")
# print(f"Final answer:\n{content}")
# print("=" * 60)

from langchain_tavily import TavilySearch

web_search = TavilySearch(max_results = 2)
# search_result = web_search.invoke("介绍一下 LangChain 这个框架")
#
# print(search_result)

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

class QAWithRetrievalArgs(BaseModel):
    query: str = Field(..., description = "用户的查询问题")

def query_retrieval_knowledge(query: str) -> str:
    """
    一个基于LangChain知识库检索的问答工具。
    专门用于回答与 LangChain 相关的技术问题。

    ⚠️ 重要：此工具仅适用于 LangChain 相关问题！
    如果问题与 LangChain 无关，请使用网络搜索工具。
    """
    # 定义 LangChain 相关关键词
    langchain_keywords = [
        'langchain', 'langgraph', 'langsmith', 'lcel',
        'chain', 'agent', 'retriever', 'embedding', 'vector',
        'rag', 'prompt', 'llm', 'chatmodel', 'runnable',
        '链', '代理', '检索器', '向量', '提示词', '模型'
    ]

    query_lower = query.lower()
    is_langchain_related = any(keyword in query_lower for keyword in langchain_keywords)

    # 如果问题不相关，提示用户使用网络搜索
    if not is_langchain_related:
        return (
            "此问题似乎与 LangChain 无关。请使用网络搜索工具(tavily_search_results_json)获取答案。\n"
            f"问题: {query}"
        )

    retrieval_chain = ensemble_retriever | format_docs
    docs = retrieval_chain.invoke(query)

    # check result length
    if not docs or len(docs.strip()) < 50:
        return (
            f"知识库中未找到关于'{query}'的充分信息。\n"
            "建议: 请使用网络搜索工具(tavily_search_results_json)获取答案。"
        )

    return docs

qa_tool = StructuredTool.from_function(
    func = query_retrieval_knowledge, # 生成基于检索的问答工具
    name = "query_retrieval_knowledge", # 工具名称
    description = (
        "🎯 专用于回答 LangChain 技术相关问题的知识库检索工具。\n"
        "适用范围：LangChain、LangGraph、LangSmith、LCEL、Agent、RAG、Retriever、Embedding、Prompt 等相关技术。\n"
        "⚠️ 限制：仅包含 LangChain 相关文档，不适用于其他领域问题（如烹饪、历史、科学等）。\n"
        "如果问题与 LangChain 无关，请使用网络搜索工具 tavily_search_results_json。"
    ),
    args_schema = QAWithRetrievalArgs, # 参数模式
    return_direct = False # 不直接返回工具结果
)

# result = qa_tool.invoke("LangChain 是什么？")
# print("QA Tool Result:\n", result)

# 定义高风险知识库敏感数据查询工具
class SensitiveKnowledgeQueryArgs(BaseModel):
    query: str = Field(description="查询的敏感主题或关键词")
    data_category: str = Field(
        description="数据类别：confidential(机密), internal(内部), sensitive(敏感)",
        default="confidential"
    )

def query_sensitive_knowledge(query: str, data_category: str = "confidential") -> str:
    """
    ⚠️ 高风险操作：基于 RAG 的敏感知识库检索

    使用向量检索 + BM25 混合检索敏感文档。
    包含机密文档、内部资料、敏感信息等。

    风险等级：🔴 高风险
    - 访问机密文档和敏感信息
    - 可能涉及商业机密、个人隐私
    - 需要权限验证和人工审核批准
    """

    print(f"\n🔴 [高风险操作] 敏感知识库 RAG 检索")
    print(f"   数据类别: {data_category}")
    print(f"   查询内容: {query}")

    sensitive_categories = {
        "confidential": "🔴 机密级",
        "internal": "🟡 内部级",
        "sensitive": "🟠 敏感级"
    }

    category_label = sensitive_categories.get(data_category, "未知级别")

    print(f"    正在检索 {category_label} 敏感文档...")
    retrieval_chain = sensitive_ensemble_retriever | format_docs
    docs = retrieval_chain.invoke(query)

    # 检查检索结果质量
    if not docs or len(docs.strip()) < 50:
        return (
            f"⚠️ 敏感知识库中未找到关于 '{query}' 的相关信息。\n"
            f"数据类别：{category_label}\n"
            f"提示：请确认查询关键词是否准确，或尝试使用不同的关键词。\n"
            f"可查询的类别：机密(confidential)、内部(internal)、敏感(sensitive)"
        )

    output = f"{category_label} 检索结果\n"
    output += "=" * 70 + "\n\n"
    output += "📋 检索到的敏感信息：\n\n"
    output += docs
    output += "\n\n" + "=" * 70
    output += f"\n\n⚠️ 安全警告：\n"
    output += f"- 以上为{category_label}信息，请妥善保管，不得外泄！\n"
    output += f"- 访问已记录，将用于安全审计\n"
    output += f"- 如需分享，请确保接收方具有相应权限\n"
    output += f"- 查询时间：{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    return output

sensitive_knowledge_tool = StructuredTool.from_function(
    func = query_sensitive_knowledge,
    name = "query_sensitive_knowledge",
    description = (
        "🔴 高风险操作：敏感知识库查询工具\n"
        "用于查询知识库中的机密文档、内部资料、敏感信息等受限数据。\n"
        "⚠️ 警告：此操作需要人工审核批准！\n"
        "适用场景：\n"
        "- 查询财务数据、战略规划等机密信息\n"
        "- 访问技术文档、人事信息等内部资料\n"
        "- 获取用户数据、客户信息等敏感数据\n"
        "安全提示：仅在必要时使用，确保有相应权限。"
    ),
    args_schema = SensitiveKnowledgeQueryArgs,
    return_direct = False
)

# result = sensitive_knowledge_tool.invoke("查询一下 2024 年 Q4 财务报告数据")
# print(f"Sensitive Knowledge Tool Result:\n{result}")

# agent execution
from typing import TypedDict
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

class Context(TypedDict):
    user_role: str

tools = [qa_tool, web_search, sensitive_knowledge_tool]

config = {"configurable": {"thread_id": "rag_test_user_001"}}

agent = create_agent(
    model = model,
    tools = tools,
    checkpointer = InMemorySaver(),
    context_schema = Context,
    debug = False
)

# for chunk in agent.stream(
#        # {"messages": [{"role": "user", "content": "Langchain 支持那些模型?"}]}, # 单工具问题
#         {"messages": [{"role": "user", "content": "比较RAG和Agentic RAG的区别，并推荐使用场景"}]}, # 复合工具问题
#         context = {"user_role": "大模型工程师"},
#         config = config,
#         stream_mode = "values"
# ):
#     if "messages" in chunk:
#         last_msg = chunk["messages"][-1]
#         if last_msg.type == "ai":
#             if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
#                 tool_call = last_msg.tool_calls[0]
#                 print(f"[AI 思考]: 决定调用工具 -> {tool_call['name']}")
#                 print(f"args: {tool_call.get('args', {})}")
#             elif last_msg.content:
#                 print(f"[AI 回答]: {last_msg.content}")

# 上下文压缩中间件 before_model
from langchain.agents.middleware import SummarizationMiddleware

summarization_middleware = SummarizationMiddleware(
    model=ChatDeepSeek(model="deepseek-chat", temperature=0.1),    # 摘要模型
    trigger = [("messages", 5),  ("tokens", 200)],
    summary_prompt="请将以下对话历史进行摘要，保留关键决策点和技术细节：\n\n{messages}\n\n摘要:"  # 摘要提示
)

# 自动工具重试中间件 wrap_tool_call
from langchain.agents.middleware import ToolRetryMiddleware
retry_middleware = ToolRetryMiddleware(
    max_retries = 3,
    tools = tools,
    retry_on = (ConnectionError, RuntimeError),
    on_failure = "return_message", # 失败后返回消息
    backoff_factor = 1.5, # 指数退避因子, 每次重试等待时间增加1.5倍
)

# Tool 调用日志中间件 (after_model)
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse,   AgentState

class ToolCallLogger:
    """工具调用日志记录器"""

    def __init__(self, log_dir: str = "LangChain_AgenticRAG/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_session_logs: List[Dict[str, Any]] = []
        self.session_start_time = datetime.now()
        self.tool_call_times: Dict[str, float] = {}  # 记录工具调用开始时间

        # Token 使用统计
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.cache_hit_tokens = 0

    def get_log_file_path(self) -> Path:
        """获取当前日志文件路径"""
        date_str = datetime.now().strftime("%Y%m%d")
        return self.log_dir / f"tool_calls_{date_str}.json"

    def log_tool_call(
        self,
        tool_name: str,
        tool_input: Any,
        tool_output: Any,
        success: bool,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        token_usage: int = 0,
    ):
        """记录单次工具调用"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_name,
            "input": str(tool_input)[:500],  # 限制长度
            "output": str(tool_output)[:1000] if success else None,
            "success": success,
            "error": error,
            "metadata": metadata or {},
            "token_usage": token_usage,
        }

        self.current_session_logs.append(log_entry)

        # 实时写入文件
        self._append_to_file(log_entry)

        # 打印日志
        status = "✅" if success else "❌"
        if not success and error:
            print(f"   Error: {error}")

    def accumulate_tokens(
        self,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        cache_hit: int = 0
    ):
        """累计 token 使用量"""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens += total_tokens
        self.cache_hit_tokens += cache_hit

        print(f"📊 [Token Usage] 输入: {input_tokens}, 输出: {output_tokens}, 总计: {total_tokens}")
        if cache_hit > 0:
            print(f"   缓存命中: {cache_hit} tokens")

    def _append_to_file(self, log_entry: Dict[str, Any]):
        """追加日志到文件"""
        log_file = self.get_log_file_path()

        # 读取现有日志
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []
        else:
            logs = []

        # 添加新日志
        logs.append(log_entry)

        # 写回文件
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.current_session_logs:
            return {"message": "No logs yet"}

        total_calls = len(self.current_session_logs)
        successful_calls = sum(1 for log in self.current_session_logs if log["success"])
        failed_calls = total_calls - successful_calls

        # 统计工具使用次数
        tool_counts = {}
        for log in self.current_session_logs:
            tool_name = log["tool_name"]
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

        return {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "success_rate": f"{(successful_calls/total_calls*100):.1f}%" if total_calls > 0 else "0%",
            "tool_usage": tool_counts,
            "token_usage": {
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_tokens,
                "cache_hit_tokens": self.cache_hit_tokens
            },
            "session_duration": str(datetime.now() - self.session_start_time)
        }

    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        print("\n" + "="*70)
        print("📊 Tool Call Statistics")
        print("="*70)
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print("="*70)


class ToolLoggingMiddleware(AgentMiddleware):
    """
    创建工具日志中间件
    使用 @wrap_model_call 装饰器从 ModelRequest 获取消息历史
    """
    def __init__(self, log_dir: str = "LangChain_AgenticRAG/logs"):
        super().__init__()
        self.logger = ToolCallLogger()


    def after_model(self,state: AgentState, runtime) -> None:
        """
        从 ModelRequest 中获取消息历史，记录工具调用信息

        Args:
            request: ModelRequest 包含 state (包括 messages)
            handler: 处理函数，执行实际的模型调用

        Returns:
            ModelResponse 模型响应
        """
        # 从 state 获取消息历史
        messages = state.get("messages", [])

        # print(f"🔍 [Tool Logging] 分析消息历史，{messages} 消息")

        # 检查消息历史中的工具调用和结果
        for msg in messages:
            # 检测 AI 消息并提取 token 使用信息
            if hasattr(msg, 'type') and msg.type == 'ai':
                # 优先从 usage_metadata 获取
                if hasattr(msg, 'usage_metadata') and msg.usage_metadata:
                    input_tokens = msg.usage_metadata.get('input_tokens', 0)
                    output_tokens = msg.usage_metadata.get('output_tokens', 0)
                    total_tokens = msg.usage_metadata.get('total_tokens', 0)

                    # 获取缓存命中信息
                    cache_hit = 0
                    if 'input_token_details' in msg.usage_metadata:
                        cache_hit = msg.usage_metadata['input_token_details'].get('cache_read', 0)

                    # 累计 token
                    self.logger.accumulate_tokens(input_tokens, output_tokens, total_tokens, cache_hit)

                # 备选：从 response_metadata 获取
                elif hasattr(msg, 'response_metadata') and msg.response_metadata:
                    token_usage = msg.response_metadata.get('token_usage', {})
                    if token_usage:
                        input_tokens = token_usage.get('prompt_tokens', 0)
                        output_tokens = token_usage.get('completion_tokens', 0)
                        total_tokens = token_usage.get('total_tokens', 0)
                        cache_hit = token_usage.get('prompt_cache_hit_tokens', 0)

                        # 累计 token
                        self.logger.accumulate_tokens(input_tokens, output_tokens, total_tokens, cache_hit)

            # 检测 AI 消息中的工具调用请求
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    # tool_call 可能是字典或对象，需要兼容两种方式
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get('name', 'unknown')
                        tool_args = tool_call.get('args', {})
                        tool_id = tool_call.get('id', 'unknown_id')
                    else:
                        tool_name = getattr(tool_call, 'name', 'unknown')
                        tool_args = getattr(tool_call, 'args', {})
                        tool_id = getattr(tool_call, 'id', 'unknown_id')

                    # 记录工具调用开始时间
                    if tool_id not in self.logger.tool_call_times:
                        self.logger.tool_call_times[tool_id] = time.time()
                        print(f"\n🔧 [Tool Logging] 检测到工具调用: {tool_name}")
                        print(f"   工具ID: {tool_id}")
                        print(f"   参数: {str(tool_args)[:200]}...")

            # 检测工具返回消息
            if hasattr(msg, 'type') and msg.type == 'tool':
                tool_name = getattr(msg, 'name', 'unknown')
                tool_content = getattr(msg, 'content', '')
                tool_call_id = getattr(msg, 'tool_call_id', 'unknown_id')
                token_usage = getattr(msg, 'token_usage', 0)

                # 判断是否成功
                success = not tool_content.startswith('❌') and not tool_content.startswith('Error')
                error_msg = tool_content if not success else None

                # 记录日志
                self.logger.log_tool_call(
                    tool_name=tool_name,
                    tool_input="[从消息历史提取]",
                    tool_output=tool_content,
                    success=success,
                    error=error_msg,
                    metadata={
                        "tool_call_id": tool_call_id,
                        "timestamp": datetime.now().isoformat(),
                        "message_type": msg.type
                    },
                    token_usage=token_usage
                )
        # 打印当前统计信息
        self.logger.print_statistics()


# 实例化日志中间件
logging_middleware = ToolLoggingMiddleware(log_dir="./logs")

from langchain.agents.middleware import ToolCallLimitMiddleware
# 工具调用限制中间件 after_model
retrieval_limit_middleware = ToolCallLimitMiddleware(
    tool_name="query_retrieval_knowledge",
    run_limit=3,  # 每次运行最多调用 3 次
    exit_behavior="continue"  # 超限后继续执行，但阻止工具调用
)

sensitive_limit_middleware = ToolCallLimitMiddleware(
    tool_name="query_sensitive_knowledge",
    run_limit=3,  # 每次运行最多调用 3 次
    exit_behavior="continue"  # 超限后继续执行，但阻止工具调用
)


# hilt 人工介入中间件 after_model
from langchain.agents.middleware import HumanInTheLoopMiddleware

official_hitl_middleware = HumanInTheLoopMiddleware(
    interrupt_on={"query_sensitive_knowledge": True},
    description_prefix="需要人工批准才能查询敏感知识库"
)

# 动态提示词中间件 wrap_model_call
from langchain.agents.middleware import dynamic_prompt

@dynamic_prompt
def rag_optimized_prompt(request: ModelRequest) -> str:
    """
    根据检索状态动态生成提示词
    核心逻辑：通过分析消息历史中的工具调用次数，确定当前所处的 RAG 阶段
    """
    messages = request.messages if hasattr(request, 'messages') else []

    # 统计所有工具调用中的知识库查询次数（包括检索和敏感查询）
    retrieval_count = 0
    for msg in messages:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                name = tool_call.name if hasattr(tool_call, 'name') else tool_call.get('name')
                # 统计知识库查询次数（包括检索和敏感查询）
                if name == 'query_retrieval_knowledge' or name == 'tavily_search_results_json' or name == 'query_sensitive_knowledge': # 通过把查询包装成工具, 统计调用工具的次数
                    retrieval_count += 1

    print(f"DEBUG: 当前累计检索次数: {retrieval_count}")

    # 基础提示词
    base_prompt = """你是一个智能知识助手，能够自主检索信息并回答问题。

    🔧 可用工具说明：
    1. query_retrieval_knowledge: 专门用于 LangChain 技术问题（LangChain、LangGraph、Agent、RAG、Retriever 等）
    2. tavily_search_results_json: 用于通用问题的网络搜索（烹饪、历史、科学、新闻等）
    3. query_sensitive_knowledge: 🔴 高风险工具 - 查询敏感知识库（财务数据、战略规划、客户信息等机密资料）

    ⚠️ 工具选择原则：
    - 如果问题涉及 LangChain 相关技术 → 使用 query_retrieval_knowledge
    - 如果问题与 LangChain 无关（如烹饪、历史、科学等） → 直接使用 tavily_search_results_json
    - 如果问题涉及敏感数据查询（财务、战略、客户、人事等） → 使用 query_sensitive_knowledge
    - 不要对非 LangChain 问题调用知识库检索工具

    🔴 高风险工具使用注意事项：
    - query_sensitive_knowledge 需要人工审核批准才能执行
    - 仅在用户明确请求查询机密/敏感信息时使用
    - 调用此工具后，系统会暂停等待管理员批准
    - 适用场景：财务报告、战略规划、客户档案、人事薪资、技术文档等

    请遵循以下流程：
    1. 分析用户问题的类型和复杂度
    2. 判断问题是否与 LangChain 相关，或是否涉及敏感数据
    3. 选择合适的检索工具
    4. 评估检索结果的质量（覆盖率、完整性、相关性）
    5. 如果结果不足，主动进行补充检索
    6. 综合所有信息生成最终回答
    """

    # 初始状态：未进行任何知识库查询
    if retrieval_count == 0:
        return base_prompt + """

        【当前状态：初始阶段】
        ⚠️ 重要：你还没有进行任何检索！

        请先判断问题类型：
        - 如果是 LangChain 相关问题 → 使用 query_retrieval_knowledge
        - 如果是其他领域问题 → 使用 tavily_search_results_json
        - 如果涉及敏感数据查询 → 使用 query_sensitive_knowledge（需人工批准）

        ❌ 禁止在没有检索的情况下直接回答问题。
        """

    # 信息评估阶段：已进行 1-2 次知识库查询
    elif retrieval_count < 3:
        return base_prompt + f"""

        【当前状态：信息评估（已检索 {retrieval_count} 次）】
        请检查上一步工具返回的搜索结果：
        1. 信息是否覆盖了用户问题的全部维度？
        2. 多个来源的信息是否一致？

        👉 决策路径：
        - 如果信息不足或有歧义 -> 请换个关键词或角度进行补充检索。
        - 如果信息已经充分 -> 请根据上下文生成最终回答。
        """

    # 最终回答阶段：已进行 3 次及以上知识库查询
    else:
        return base_prompt + f"""

        【当前状态：最终回答（已检索 {retrieval_count} 次）】
        🛑 已达到最大检索次数限制，请停止检索！

        请必须基于当前已有的所有信息，生成最终的回答。
        如果检索到的信息仍不能完全回答问题，请诚实地说明信息的局限性或缺失部分。
        """

# 中间件集合
middlewares = [
    # before_model: 准备阶段，上下文压缩中间件
    summarization_middleware,

    # wrap_model_call: 模型调用包裹，智能切换系统提示词
    rag_optimized_prompt,

    # after_model: 后处理（逆序执行，所以倒着写）
    official_hitl_middleware,  # 最后执行：人工审核（可能中断）
    logging_middleware,  # 倒数第二：记录日志
    sensitive_limit_middleware,  # 倒数第三：限制敏感工具
    retrieval_limit_middleware,  # 最先执行：限制检索工具

    # wrap_tool_call: 工具调用包裹
    retry_middleware,
]

from typing import TypedDict
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# 创建并运行 Agent
class Context(TypedDict):
    user_role: str

config = {
    "configurable": {"thread_id": "rag_test_user_final"}
}

agent = create_agent(
    tools = tools,
    model = model,
    middleware = middlewares,
    debug = False,
    checkpointer = InMemorySaver(),
    context_schema = Context
)

# 触发 hitl 中间件
from langgraph.types import Command
# 导入 HITL 相关类
from langchain.agents.middleware.human_in_the_loop import (
    HITLResponse,
    ApproveDecision,
    EditDecision,
    RejectDecision
)

def run_hitl_interactive_test():
    """
    运行人工介入中间件测试交互会话
    参考 HITL_demo.py
    """
    print("\n" + "=" * 70)
    print("🚀 开始执行 Agentic RAG 测试 (HITL 人工干预模式)")
    print("=" * 70)

    # 测试提示词：触发敏感知识库查询
    user_input = "帮我查询一下2024年Q4财务报告数据的详细内容。"
    print(f"\n[用户]: {user_input}")

    print("\n[系统]: 开始处理请求...")

    for event in agent.stream(
        {
            "messages": [{"role": "user", "content": user_input}]
        },
        config = config,
        stream_mode = "values",
        context = {"user_role": "财务分析师"}
    ):
        if "messages" in event:
            last_msg = event["messages"][-1]
            if last_msg.type == "ai" and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                print(f"[AI 决策]: 准备调用工具 -> {last_msg.tool_calls[0]['name']}")

    # 观察中断状态
    snapshot = agent.get_state(config)

    print(f"\n--- 🛑 执行已暂停 (HITL Middleware 触发) ---")
    print(f"下一步骤: {snapshot.next}")
    print(f"任务数量: {len(snapshot.tasks) if snapshot.tasks else 0}")

    if snapshot.tasks:
        last_message = snapshot.values["messages"][-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_call = last_message.tool_calls[0]

            print(f"\n{'=' * 70}")
            print("🔴 检测到高风险操作：敏感知识库查询")
            print(f"{'=' * 70}")
            print(f"工具名称: {tool_call['name']}")
            print(f"查询内容: {tool_call['args'].get('query', 'N/A')}")
            print(f"数据类别: {tool_call['args'].get('data_category', 'confidential')}")
            print(f"{'=' * 70}")

            # === 第三步：人工决策 ===
            approval = input(
                "\n[管理员]: 是否批准此操作? (y/n/e[编辑]): ").strip().lower()

            if approval == 'y':
                # === 批准操作 ===
                print("\n[系统]: ✅ 操作已批准，继续执行...")

                # 继续执行代理
                hitl_response = HITLResponse(
                    decisions = [ApproveDecision(type = "approve")]  # 批准决策
                )

                for event in agent.stream(
                    Command(resume = hitl_response),
                    config = config,
                    stream_mode = "values"
                ):
                    if "messages" in event:
                        last_msg = event["messages"][-1]
                        if last_msg.type == "tool":
                            print(f"[工具返回]: {last_msg.content}")
                        elif last_msg.type == "ai" and last_msg.content:
                            print(f"[AI 最终回答]: {last_msg.content}")

            elif approval == 'e':
                # === 编辑操作 ===
                print("\n[系统]: ✏️  编辑模式...")
                print(f"当前参数: {tool_call['args']}")

                new_query = input(f"current query [{tool_call['args'].get('query', '')}], enter new query or press Enter to keep: ").strip()
                new_category = input(f"current data_category [{tool_call['args'].get('data_category', 'confidential')}], enter new category or press Enter to keep: ").strip()

                updated_args = tool_call['args'].copy()
                if new_query:
                    updated_args['query'] = new_query
                if new_category:
                    updated_args['data_category'] = new_category

                print(f"\n[系统]: 使用更新后的参数继续执行...")
                print(f"更新后的参数: {updated_args}")

                hitl_response = HITLResponse(
                    decisions = [
                        EditDecision(
                            type = "edit",
                            edited_action = {
                                "name": tool_call['name'],
                                "args": updated_args
                            }
                        )
                    ]
                )

                for event in agent.stream(
                        Command(resume = hitl_response),
                        config = config,
                        stream_mode = "values"
                ):
                    if "messages" in event:
                        last_msg = event["messages"][-1]
                        if last_msg.type == "tool":
                            print(f"\n[工具输出]:\n{last_msg.content}")
                        elif last_msg.type == "ai" and last_msg.content:
                            print(f"\n[AI 最终回复]: {last_msg.content}")

            else:
                # === 拒绝操作 ===
                print("\n[系统]: ❌ 操作被拒绝")

                rejection_reason = input(
                    "拒绝原因 (可选): ").strip() or "操作被管理员拒绝，权限不足"

                hitl_response = HITLResponse(
                    decisions = [RejectDecision(
                        type = "reject",
                        message = rejection_reason
                    )]
                )

                for event in agent.stream(
                        Command(resume = hitl_response),
                        config = config,
                        stream_mode = "values"
                ):
                    if "messages" in event:
                        last_msg = event["messages"][-1]
                        if last_msg.type == "ai" and last_msg.content:
                            print(f"\n[AI 回复]: {last_msg.content}")
                        elif last_msg.type == "tool":
                            print(f"\n[工具消息]: {last_msg.content}")

                print("\n[系统]: 流程已终止")

        else:
            print("⚠️  没有检测到待处理的工具调用")
    else:
        print("ℹ️  流程已完成，没有触发中断")
        if snapshot.values.get("messages"):
            last_msg = snapshot.values["messages"][-1]
            if last_msg.type == "ai" and last_msg.content:
                print(f"\n[最终回复]: {last_msg.content}")

    print("\n" + "=" * 70)
    print("✅ HITL 测试完成！")
    print("=" * 70)

    # 打印统计信息
    print("\n📊 中间件统计信息:")
    logging_middleware.logger.print_statistics()

#run_hitl_interactive_test()


#%%
def run_normal_rag_test():
    """
    运行普通 RAG 检索测试
    测试 query_retrieval_knowledge 工具的检索流程
    """
    print("\n" + "="*70)
    print("🚀 开始执行普通 RAG 检索测试")
    print("="*70)

    # 测试提示词：触发 LangChain 知识库检索
    test_queries = [
        "LangChain 中的 Agent 是什么？它有哪些核心组件？",
        "如何在 LangChain 中使用 RAG 进行问答？",
        "LangGraph 和 LangChain 有什么区别？"
    ]

    print("\n可用的测试问题：")
    for i, query in enumerate(test_queries, 1):
        print(f"{i}. {query}")

    choice = input("\n请选择测试问题 (1-3) 或输入自定义问题: ").strip()

    if choice.isdigit() and 1 <= int(choice) <= len(test_queries):
        user_input = test_queries[int(choice) - 1]
    else:
        user_input = choice if choice else test_queries[0]

    print(f"\n[用户]: {user_input}")
    print("\n[系统]: 开始处理请求...\n")

    # 使用新的 thread_id 避免与 HITL 测试冲突
    rag_config = {"configurable": {"thread_id": "rag-test-thread"}}

    # 用于跟踪已打印的消息，避免重复
    printed_message_ids = set()

    # 执行 Agent 流程
    for event in agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config=rag_config,
        stream_mode="values",
        context={"user_role": "开发者"}
    ):
        if "messages" in event:
            last_msg = event["messages"][-1]

            # 使用消息 ID 来避免重复打印
            msg_id = getattr(last_msg, 'id', None)
            if msg_id and msg_id in printed_message_ids:
                continue

            if msg_id:
                printed_message_ids.add(msg_id)

            # 显示 AI 的思考过程
            if last_msg.type == "ai":
                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    tool_call = last_msg.tool_calls[0]
                    print(f"🤖 [AI 决策]: 调用工具 -> {tool_call['name']}")
                    print(f"   参数: {tool_call.get('args', {})}")
                elif last_msg.content:
                    print(f"\n💬 [AI 回复]:\n{last_msg.content}")

            # 显示工具执行结果
            elif last_msg.type == "tool":
                tool_name = getattr(last_msg, 'name', 'unknown')
                print(f"\n🔧 [工具执行]: {tool_name}")
                print(f"📄 [检索结果]:\n{'-'*70}")
                # 只显示前500个字符，避免输出过长
                content = last_msg.content
                if len(content) > 500:
                    print(f"{content[:500]}...\n(结果已截断，共 {len(content)} 字符)")
                else:
                    print(content)
                print(f"{'-'*70}\n")

    print("\n" + "="*70)
    print("✅ 普通 RAG 检索测试完成！")
    print("="*70)

    # 打印统计信息
    print("\n📊 中间件统计信息:")
    logging_middleware.logger.print_statistics()

run_normal_rag_test()



