from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MathServer")

# 加减乘除

@mcp.tool()
def add(a: float, b: float) -> float:
    """计算两个数的和"""
    return a + b

@mcp.tool()
def subtract(a: float, b: float) -> float:
    """计算两个数的差"""
    return a - b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """计算两个数的积"""
    return a * b

@mcp.tool()
def divide(a: float, b: float) -> float:
    """计算两个数的商"""
    if b == 0:
        raise ValueError("除数不能为 0")
    return a / b

if __name__ == "__main__":
    mcp.run(transport = "stdio")