from datetime import datetime
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from db.retriever import lookup_policy
from tools.flight_tools import search_flights
from langgraph.graph import END
from langchain_core.runnables import Runnable
from assistants.base import Assistant
from assistants.common import State
from langgraph.prebuilt import tools_condition


class ToFlightBookingAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle flight updates and cancellations."""

    request: str = Field(
        description="Any necessary followup questions the update flight assistant should clarify before proceeding."
    )


class ToHotelBookingAssistant(BaseModel):
    """Transfer work to a specialized assistant to handle hotel bookings."""

    location: str = Field(
        description="The location where the user wants to book a hotel."
    )
    checkin_date: str = Field(description="The check-in date for the hotel.")
    checkout_date: str = Field(description="The check-out date for the hotel.")
    request: str = Field(
        description="Any additional information or requests from the user regarding the hotel booking."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "location": "Zurich",
                "checkin_date": "2023-08-15",
                "checkout_date": "2023-08-20",
                "request": "I prefer a hotel near the city center with a room that has a view.",
            }
        }


class ToBookCarRental(BaseModel):
    """Transfers work to a specialized assistant to handle car rental bookings."""

    location: str = Field(
        description="The location where the user wants to rent a car."
    )
    start_date: str = Field(description="The start date of the car rental.")
    end_date: str = Field(description="The end date of the car rental.")
    request: str = Field(
        description="Any additional information or requests from the user regarding the car rental."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "location": "Basel",
                "start_date": "2023-07-01",
                "end_date": "2023-07-05",
                "request": "I need a compact car with automatic transmission.",
            }
        }


class ToBookExcursion(BaseModel):
    """Transfers work to a specialized assistant to handle excursion bookings."""

    location: str = Field(
        description="The location where the user wants to book an excursion."
    )
    date: str = Field(description="The date for the excursion.")
    request: str = Field(
        description="Any additional information or requests from the user regarding the excursion booking."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "location": "Interlaken",
                "date": "2023-09-10",
                "request": "I'm interested in outdoor activities like hiking or paragliding.",
            }
        }


# 主助手工具
primary_assistant_tools = [
    TavilySearchResults(max_results=1),
    search_flights,
    lookup_policy,
]


# 主助手提示模板
primary_assistant_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful customer support assistant for Swiss Airlines. "
                "Your primary role is to search for flight information and company policies to answer customer queries. "
                "If a customer requests to update or cancel a flight, book a car rental, book a hotel, or get trip recommendations, "
                "delegate the task to the appropriate specialized assistant by invoking the corresponding tool. You are not able to make these types of changes yourself."
                " Only the specialized assistants are given permission to do this for the user."
                "The user is not aware of the different specialized assistants, so do not mention them; just quietly delegate through function calls. "
                "Provide detailed information to the customer, and always double-check the database before concluding that information is unavailable. "
                " When searching, be persistent. Expand your query bounds if the first search returns no results. "
                " If a search comes up empty, expand your search before giving up."
                "\n\nCurrent user flight information:\n<Flights>\n{user_info}\n</Flights>"
                "\nCurrent time: {time}.",
            ),
            ("placeholder", "{messages}"),
        ]
    ).partial(time=datetime.now)


def route_primary_assistant(
    state: State,
):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == ToFlightBookingAssistant.__name__:
            return "enter_update_flight"
        elif tool_calls[0]["name"] == ToBookCarRental.__name__:
            return "enter_book_car_rental"
        elif tool_calls[0]["name"] == ToHotelBookingAssistant.__name__:
            return "enter_book_hotel"
        elif tool_calls[0]["name"] == ToBookExcursion.__name__:
            return "enter_book_excursion"
        return "primary_assistant_tools"
    raise ValueError("Invalid route")

def create_primary_assistant(llm: Runnable,):
    """创建主助手"""
    assistant_runnable = primary_assistant_prompt | llm.bind_tools(
        [
            ToFlightBookingAssistant,
            ToHotelBookingAssistant,
            ToBookCarRental,
            ToBookExcursion,
        ]
        + primary_assistant_tools
    )
    return Assistant(assistant_runnable)