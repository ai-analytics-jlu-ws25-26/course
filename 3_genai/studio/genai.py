import operator
from typing import Annotated, List, Optional, TypedDict
import numpy as np

import torch
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

# Vector Storage
vector_store = FAISS.load_local(
    "./faiss_vector_store",
    embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
    ),
    allow_dangerous_deserialization=True
)

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Web search tool
tavily_search = TavilySearchResults(max_results=2)

dax_40 = [
    'adidas', 'airbus', 'allianz', 'basf', 'bayer', 'beiersdorf', 'bmw',
    'brenntag', 'commerzbank', 'continental', 'daimler-truck',
    'deutsche-bank', 'deutsche-boerse', 'deutsche-post',
    'deutsche-telekom', 'eon', 'fresenius', 'fresenius-medical-care',
    'hannover-rück', 'heidelberg-materials', 'henkel',
    'infineon-technologies', 'mercedes-benz', 'merck',
    'mtu', 'münchener-rück', 'porsche',
    'qiagen', 'rheinmetall', 'rwe', 'sap', 'sartorius', 'siemens',
    'siemens-energy', 'siemens-healthineers', 'symrise',
    'volkswagen', 'vonovia', 'zalando']


class Topic(BaseModel):
    dax_40_related: bool = Field(description="Whether the query is related to DAX 40 companies.")
    topic: str = Field(description="Topic of the query.")


class Company(BaseModel):
    name: str = Field(description="Name of DAX 40 company.")


class Companies(BaseModel):
    companies: Optional[List[Company]] = Field(description="List of DAX 40 companies.")


class OverallState(MessagesState):
    topic: Topic
    companies: Companies
    context: Annotated[List[str], operator.add]
    context_amount: Annotated[List[int], operator.add]
    context_count: int
    final_answer: str


class AnnualReportState(TypedDict):
    company: Company
    context_report: Annotated[List[str], operator.add]


topic_extraction_instruction = """You are part of an AI agent 
designed to answer questions about the risks DAX 40 companies
are facing.

Your task is to judge whether the following user query is a 
question concerned about DAX 40 companies or not.
Note that the user query may not explicitly mention any DAX 40 companies, 
but it may still be related to them. If it mentions any DAX 40 companies, 
it is for sure a DAX 40 related question.
Questions that refer to any type of risks that corporations 
could face should be flagged as DAX 40 related.

If the user query is related to DAX 40 companies, you should 
extract the topic of the question. If the user query is not 
related to DAX 40 companies, please return nothing.
The topic should be a short phrase that summarizes the main subject 
of the question. Please make sure to retain specific keywords that 
are relevant to the topic.

This is the user query, placed between triple backticks, from which 
you should extract the topic: ``` {message} ```.

Company names or the term 'DAX 40' should not be included in the topic.

# Few-shot examples:

User query: "How is Siemens responding to supply chain disruptions?"
Output:
  "dax_40_related": true,
  "topic": "supply chain disruptions"

User query: "What are the latest trends in renewable energy adoption among major German companies?"
Output:
  "dax_40_related": true,
  "topic": "renewable energy adoption"

User query: "What is the weather like in Berlin today?"
Output:
  "dax_40_related": false,
  "topic": None
"""


def extract_topic_node(state: OverallState):

    messages = state.get('messages')
    last_message = messages[-1].content

    # Enforce structured output
    structured_llm = llm.with_structured_output(Topic)

    # Adding last_message to the system prompt
    system_message = topic_extraction_instruction.format(message=last_message)

    # Extract topic
    topic = structured_llm.invoke(
        [SystemMessage(content=system_message)])

    # Return topic which is added to the state
    return {'topic': topic}


def general_router(state: OverallState):

    topic = state.get('topic')
    dax_40_related = topic.dax_40_related
    
    # Check if the query is related to DAX 40 companies
    if dax_40_related:
        return "on-topic"
    else:
        return "off-topic"


def off_topic_response_node(state: OverallState):

    final_answer = """Your question is not concerned about DAX 40 companies 
    which is why I cannot provide any answer to you. Please ask a question
    related to the risks DAX 40 companies are facing."""

    return {"final_answer": final_answer}


company_extraction_instruction = """You are tasked with analyzing whether the following 
user query relates to any specific DAX 40 companies: ```{message}```.

If you find specific DAX 40 companies mentioned in the user query, 
map them to the ones mentioned in the following list: ```{dax_40}```.

It may well be that the user query is a more generic question 
that is not related to any specific company. In this case return None.

# Few-shot examples:

User query: "What impact does the new emission regulation have on Volkswagen and BMW?"
Output:
["volkswagen", "bmw"]

User query: "How is Siemens adapting to the global chip shortage?"
Output:
["siemens"]

User query: "What are the risks facing the German automotive industry this year?"
Output:
None
"""

def extract_companies_node(state: OverallState, dax_40: list = dax_40):

    messages = state.get('messages')
    last_message = messages[-1].content
    dax_40 = ', '.join(dax_40)
    
    structured_llm = llm.with_structured_output(Companies)
    system_message = company_extraction_instruction.format(
        message=last_message, dax_40=dax_40)
    companies = structured_llm.invoke(
        [SystemMessage(content=system_message)])

    return {'companies': companies.companies}


def vector_store_router(state: OverallState, dax_40: list = dax_40):

    companies = state.get('companies', None)
    topic =  state.get('topic')

    # Check if any companies were extracted
    if companies is not None:
        return [Send(
            "vector_store", 
            {"company": c, "topic": topic.topic}) for c in companies]
    
    else:
        return [Send(
            "vector_store", 
            {"company": Company(name=c), "topic": topic.topic}) for c in dax_40]


single_answer_generation_instruction = """Based on the following context, denoted within triple tickmarks, for 
{company}, generate an answer to the topic {topic}: ``` {context} ```

Try your best to generate a answer given the above context!

Only if the context does not provide any information about {topic}, please answer that 
the annual report of {company} does not provide any information 
about the topic {topic}.
"""


def vector_store_node(
        state: AnnualReportState, 
        vector_store: FAISS = vector_store):

    company = state.get('company')
    topic = state.get('topic')

    context_report = vector_store.similarity_search(
        query=topic, 
        k=2, 
        filter={"company": company.name})
    context_amount = len(context_report)

    if context_amount == 0:
        answer = f"""The annual report of {company.name} does not provide any 
        information about the topic '{topic}'."""
    
    else:
        system_message = single_answer_generation_instruction.format(
            company=company.name, 
            topic=topic, 
            context='\n\n---\n\n'.join(
                [node.page_content for node in context_report]))
        answer = llm.invoke(system_message).content
    
    return {"context_report": context_report, "context": [answer],
            "context_amount": [context_amount]}


def count_context_node(state: OverallState):

    context_amount = state.get('context_amount')
    state['context_count'] = np.sum(context_amount)

    return {"context_count": state['context_count']}


def web_search_router(state: OverallState):

    context_count = state.get('context_count')

    if context_count == 0:
        return "relevant-context-not-found"
    else:
        return "relevant-context-found"


def web_search_node(state: OverallState):

    messages = state.get('messages')
    last_message = messages[-1].content
    
    search_docs = tavily_search.invoke(last_message)

    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}">\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}


final_answer_generation_instruction = """You are tasked to provide a concise 
answer to the following query: ```{message}```.

To respond to the user, you are supposed to use the following contextual 
information: ```{context}```.

If you find that the context contains repetitive information, please 
summarize it accordingly.

If the context mentions DAX 40 companies, please make sure to explicitly 
include them in your answer.
"""


def generate_answer_node(state: OverallState):
    
    messages = state.get('messages')
    last_message = messages[-1].content
    context = state.get('context', None)

    system_message = final_answer_generation_instruction.format(
        message=last_message, 
        context='\n\n---\n\n'.join(context))
    final_answer = llm.invoke(system_message).content

    # Return the company-specific answer
    return {"final_answer": final_answer}


builder = StateGraph(OverallState)

builder.add_node("extract_topic", extract_topic_node)
builder.add_node("off_topic_response", off_topic_response_node)
builder.add_node("extract_companies", extract_companies_node)
builder.add_node("vector_store", vector_store_node)
builder.add_node("count_context", count_context_node)
builder.add_node("web_search", web_search_node)
builder.add_node("generate_answer", generate_answer_node)

builder.add_edge(START, "extract_topic")
builder.add_conditional_edges("extract_topic", general_router, {
    "on-topic": "extract_companies", "off-topic": "off_topic_response"})
builder.add_conditional_edges("extract_companies", vector_store_router, {
    "map-reduce": "vector_store"})
builder.add_edge("vector_store", "count_context")
builder.add_conditional_edges("count_context", web_search_router, {
    "relevant-context-found": "generate_answer", 
    "relevant-context-not-found": "web_search"})
builder.add_edge("web_search", "generate_answer")
builder.add_edge("generate_answer", END)
builder.add_edge("off_topic_response", END)

graph = builder.compile()
