from typing import Union, List
from dotenv import load_dotenv
from langchain.agents import Tool, tool
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain.tools.render import render_text_description
from langchain.agents.output_parsers.react_single_input import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents.format_scratchpad import format_log_to_str


load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """
    Returns the length of the text.
    """
    print(f"get_text_length enter with {text=}")
    # Stripping away non aplhabetic characters
    text = text.strip("'\n").strip('"')

    return len(text)

def find_tool_by_name(tools: List[Tool], tool_name: str)-> Tool:
    """
    Finds a tool by its name.
    """
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found.")


if __name__ == "__main__":
    
    tools = [get_text_length]


    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

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

    Question: {input}
    Thought: {agent_scratchpad}
    """

    tool_names = ", ".join([tool.name for tool in tools])
    tools_description = render_text_description(tools)
    
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=tools, tool_names=tool_names)
    
    llm = ChatOllama(model="deepseek-r1:8b", temperature=0.0, stop=["\nObservation"])
    intermediate_steps = []

    agent = ({
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])} | prompt | llm | ReActSingleInputOutputParser() )

    sample_text = "What is the length in characters of the text 'kajs sadjkj ssdsdsalkjkl j'?"
    
    # agent_step might be either: An action the agent wants to take, or A finished answer.
    agent_step: Union[AgentAction, AgentFinish] = agent.invoke({"input": sample_text,
                                                                "agent_scratchpad": intermediate_steps})
    print(agent_step)


    # response = agent.invoke({"input": sample_text})
    # print("Response from agent:")
    # print(response)
    

    if isinstance(agent_step, AgentAction):
        print(f"\n\n\n agent_step is instance AgentAction\n\n\n")
        # This part does not work, should only get the name of the func, to be used later
        tool_name = agent_step.tool
        
        print(f"Tool name is: {tool_name}")
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input

        observation = tool_to_use.func(str(tool_input))
        
        print(f"Observation {observation}")
        intermediate_steps.append((agent_step, str(observation)))
    
    # agent_step might be either: An action the agent wants to take, or A finished answer.
    agent_step: Union[AgentAction, AgentFinish] = agent.invoke({"input": sample_text,
                                                                "agent_scratchpad": intermediate_steps})
    print(agent_step)

    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)