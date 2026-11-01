from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langsmith import traceable
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "qwen3:latest"

# Tools

@tool
def get_product_price(product: str) -> float:
    """Lookup price of product in catalog """
    print(f"Executing get_product_price with product: {product}")
    price = {"laptop": 1000, "phone": 500, "tablet": 750}
    return price.get(product, 0.0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """ APPLY DISCOUNT TO PRICE 
    Available discount tiers: bronze (5%), silver (10%), gold (23%) """
    print(f"Executing apply_discount with price: {price} and discount_tier: {discount_tier}")
    discount_percentage = {"bronze": 5, "silver": 10, "gold": 23}
    return round(price * (1 - discount_percentage[discount_tier]/100), 2)

#  Agent loop

@traceable(name="Langchain Agent Loop")
def run_agent(question: str) -> str:
    """Run the agent with the given question."""
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Running agent with question: {question}")
    print('='*80)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question)
    ]
    for iteration in range(1, MAX_ITERATIONS):
        print(f"-------------------  Iteration {iteration} ------------------------")
        ai_message  = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        if not tool_calls:
            print(f"Final Agent response: {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_id = tool_call.get("id")

        print(f"Selected tool: {tool_name} with args: {tool_args} and id: {tool_id}")
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            print(f"Error: Tool {tool_name} not found.")
            return "Error: Tool not found."
        observation = tool_to_use.invoke(tool_args)
        print(f"Tool observation: {observation}")

        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_id))


    print("end of agent loop")

if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)!")
    run_agent("What is the price of a laptop after applying a silver discount?")
