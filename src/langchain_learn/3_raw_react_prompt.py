# from langchain.agents import create_agent
# from langchain.tools import tool
# from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
# from langchain_ollama import ChatOllama
# from pydantic import BaseModel, Field
# from langchain.chat_models import init_chat_model

import inspect
import re

from langsmith import traceable
from dotenv import load_dotenv
import ollama

load_dotenv()

MAX_ITERATIONS = 3
MODEL = "qwen3:latest"

# Tools

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Lookup price of product in catalog """
    print(f"Executing get_product_price with product: {product}")
    price = {"laptop": 1000, "phone": 500, "tablet": 750}
    return price.get(product, 0.0)

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """ APPLY DISCOUNT TO PRICE 
    Available discount tiers: bronze (5%), silver (10%), gold (23%) """
    print(f"Executing apply_discount with price: {price} and discount_tier: {discount_tier}")
    discount_percentage = {"bronze": 5, "silver": 10, "gold": 23}
    return round(price * (1 - discount_percentage[discount_tier]/100), 2)

tools = {"get_product_price":get_product_price, "apply_discount":apply_discount}

def get_tool_descriptions(tools_dict):
    description = []
    for tool_name, tool_func in tools_dict.items():
        original_func = getattr(tool_func, "__wrapped__", tool_func)
        signature = inspect.signature(original_func)
        docstring = inspect.getdoc(original_func) or ""
        print(f"Tool: {tool_name}\nSignature: {signature}\nDocstring: {docstring}\n")
        description.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(description)

tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())
pass

react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, "ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""

@traceable(name="ollama chat")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)

#  Agent loop

@traceable(name="Langchain Agent Loop")
def run_agent(question: str) -> str:
    """Run the agent with the given question."""

    prompt = react_prompt.format(question=question)
    print(f"Running agent with question: {question}")
    scratchpad = ""

    print('='*80)
    
    for iteration in range(1, MAX_ITERATIONS):
        print(f"-------------------  Iteration {iteration} ------------------------")
        full_prompt = prompt + scratchpad
        response  = ollama_chat_traced(MODEL, messages=[{"role": "user", "content": full_prompt}], options={"stop": ["\nObservation"], "temperature": 0})
        output = response.message.content
        
        print(f"  [Parsing] Looking for Final Answer in LLM output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"  [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer
        
        tool_name = output.split("Action: ")[-1].split("\n")[0].strip()
        tool_args = output.split("Action Input: ")[-1].split("\n")[0].strip()

        if not tool_name or not tool_args:
            print("Error: Tool name or arguments not found in the output.")
            return "Error: Tool name or arguments not found."

        print(f"Selected tool: {tool_name}with args: {tool_args}")
        tool_to_use = tools.get(tool_name)
        if tool_to_use is None:
            print(f"Error: Tool {tool_name} not found.")
            observation = f"Error: Tool: '{tool_name}' not found. Available tools: {', '.join(tools.keys())}"
        else:
            observation = tool_to_use(tool_args) 

        print(f"Tool observation: {observation}")

        scratchpad =f"{output}\nObservation: {observation}\nThought:"
        pass

    print("end of agent loop")

if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)!")
    run_agent("What is the price of a laptop after applying a silver discount?")
