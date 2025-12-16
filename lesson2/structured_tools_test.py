# 这是最常用的方式，通过函数直接创建结构化工具，支持同步和异步双重实现。

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

class DivideInput(BaseModel):
    # 除法工具输入参数
    dividend: float = Field(description="被除数")
    divisor: float = Field(description="除数, 不能为 0")

def divide(dividend: float, divisor: float) -> float:
    """执行除法运算"""
    if divisor == 0:
        raise ValueError("除数不能为 0")
    return dividend / divisor

division_tool = StructuredTool.from_function(
    func = divide,
    name = "DivisionTool",
    description = "执行除法运算的工具, 自动处理除数为 0 的情况。",
    args_schema = DivideInput,
    return_direct = False # 是否直接返回公爵级过, 不经过模型再次处理
)

try:
    division_tool.invoke({"a": 10, "b": 2}) # 使用错误的参数名称
except Exception as e:
    print("Error during tool invocation:", e)

try:
    result = division_tool.invoke({"dividend": 10, "divisor": 0}) # 正确的参数名称但除数为0
except Exception as e:
    print("Error during tool invocation:", e)

# 正确调用
result = division_tool.invoke({"dividend": 10, "divisor": 2})
print("Division Result:", result)