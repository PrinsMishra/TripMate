import os 
import certifi
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import TypedDict, Annotated
import operator
import uuid

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")


# =========================
# LLM
# =========================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    max_tokens=1536,
    max_retries=5,
)

class TravelRequestPlan(BaseModel):
    needs_flights: bool = Field(description="Search flights only if the user asks for flight information.")
    needs_hotels: bool = Field(description="Search hotels only if the user asks for accommodation.")
    needs_places: bool = Field(description="Search attractions only if the user asks for places or activities.")
    needs_itinerary: bool = Field(description="Create a day-by-day itinerary only if the user asks for a trip plan.")


request_planner = llm.with_structured_output(TravelRequestPlan)


# =========================
# State
# =========================

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    places_results: str
    itinerary: str
    llm_calls: int
    needs_flights: bool
    needs_hotels: bool
    needs_places: bool
    needs_itinerary: bool


# =========================
# Request Router
# =========================

def request_router(state: TravelState):
    plan = request_planner.invoke([
        SystemMessage(content=(
            "Classify which travel-planning actions are needed for the user's exact request. "
            "Do not select flights, hotels, or places unless the user asks for them or they "
            "are clearly needed. A flight-only question selects only flights. A hotel-only "
            "question selects only hotels. A day-by-day itinerary selects places and an "
            "itinerary, but not flights or hotels unless requested. A complete-trip request "
            "that asks for flights, accommodation, and sightseeing selects all of them. "
            "Do not select an itinerary for a simple information lookup."
        )),
        HumanMessage(content=state["user_query"]),
    ])

    return {
        "needs_flights": plan.needs_flights,
        "needs_hotels": plan.needs_hotels,
        "needs_places": plan.needs_places,
        "needs_itinerary": plan.needs_itinerary,
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Selective Research
# =========================

def research_agent(state: TravelState):
    query = state["user_query"]
    results = {}

    if state["needs_flights"]:
        results["flight_results"] = search_flights(query)

    if state["needs_hotels"]:
        results["hotel_results"] = tavily_search(f"Best hotels for {query}")

    if state["needs_places"]:
        results["places_results"] = tavily_search(
            f"Best attractions and things to do for {query}"
        )

    results["messages"] = [AIMessage(content="Requested travel research completed.")]
    return results

# =========================
# Itinerary Agent
# =========================

def itinerary_agent(state: TravelState):
    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Flight Results:
{state.get('flight_results', '') or 'Not requested.'}

Hotel Results:
{state.get('hotel_results', '') or 'Not requested.'}

Places and Activities:
{state.get('places_results', '') or 'Not requested.'}

Make the itinerary practical, budget-aware, and easy to follow.
"""

    response = llm.invoke([
        SystemMessage(content="You are an expert travel planner."),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }



# =========================
# Final Response Agent
# =========================

def final_agent(state: TravelState):
    final_prompt = f"""
Generate the final travel response for the user.

User Request:
{state['user_query']}

Flights:
{state.get('flight_results', '') or 'Not requested.'}

Hotels:
{state.get('hotel_results', '') or 'Not requested.'}

Places and Activities:
{state.get('places_results', '') or 'Not requested.'}

Itinerary:
{state.get('itinerary', '') or 'Not requested.'}

Important:
- Answer only the user's request; do not add unrequested hotel, flight, or itinerary sections.
- Use only the provided research for factual claims. Do not invent prices, schedules, availability, or search results.
- If a requested flight price or date-specific result is absent from the research, say it is unavailable.
- Clearly distinguish suggestions from verified bookings or availability.
"""

    response = llm.invoke([
        SystemMessage(content="You are a professional AI travel booking assistant."),
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Build Graph
# =========================

graph = StateGraph(TravelState)

graph.add_node("request_router", request_router)
graph.add_node("research_agent", research_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "request_router")
graph.add_edge("request_router", "research_agent")
graph.add_conditional_edges(
    "research_agent",
    lambda state: "itinerary_agent" if state["needs_itinerary"] else "final_agent",
    {
        "itinerary_agent": "itinerary_agent",
        "final_agent": "final_agent",
    },
)
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)


# =========================
# PostgreSQL Checkpointer
# =========================
DATABASE_URL = get_database_url()

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row
)

checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)



# =========================
# Function for FastAPI
# =========================

def run_travel_agent(user_input: str, thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "places_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )

    final_answer = result["messages"][-1].content

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "places_results": result.get("places_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0),
    }
