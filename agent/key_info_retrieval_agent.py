from langchain import hub
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_openai import ChatOpenAI

from tools.retrieval_qa_with_llm_and_resort_tool import RetrievalQAWithLLMAndResortTool


def get_key_info_retrieval_agent(model: str = 'gpt-3.5-turbo', temperature: float = 0.5):
    # Get the prompt to use - you can modify this!
    prompt = hub.pull("hwchase17/openai-tools-agent")

    # Choose the LLM that will drive the agent
    # Only certain models support this
    llm = ChatOpenAI(model=model, temperature=temperature)
    tools = [RetrievalQAWithLLMAndResortTool()]
    # Construct the OpenAI Tools agent
    agent = create_openai_tools_agent(llm, tools, prompt)

    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor
