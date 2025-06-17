from dotenv import load_dotenv
from langchain.agents import tool
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain.tools.render import render_text_description

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """
    Returns the length of the text.
    """
    print(f"get_text_length enter with {text=}")
    # Stripping away non aplhabetic characters
    text = text.strip("'\n").strip('"')

    return 2*len(text)


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
    Thought:
    """

    tool_names = ", ".join([tool.name for tool in tools])
    tools_description = render_text_description(tools)
    
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=tools, tool_names=tool_names)
    
    llm = ChatOllama(model="mistral", temperature=0.0, stop=["\nObservation"])

    agent = prompt | llm

    sample_text = "What is the length of the text 'skjdhfsdfhjksdfjkhldsfjklhñ!'?"
    response = agent.invoke({"input": sample_text})
    print(response)

    # print(f"The length of the text is: {length}")