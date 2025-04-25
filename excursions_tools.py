import sqlite3
from datetime import date, datetime
from typing import Optional, Union
from langchain_core.tools import tool
from db import db


@tool
def search_trip_recommendations(
    location: Optional[str] = None,
    name: Optional[str] = None,
    keywords: Optional[str] = None,
) -> list[dict]:
    """
    Search for trip recommendations based on location, name, and keywords.

    Args:
        location (Optional[str]): The location of the trip recommendation. Defaults to None.
        name (Optional[str]): The name of the trip recommendation. Defaults to None.
        keywords (Optional[str]): The keywords associated with the trip recommendation. Defaults to None.

    Returns:
        list[dict]: A list of trip recommendation dictionaries matching the search criteria.
    """
    base_query = "SELECT * FROM trip_recommendations WHERE 1=1"
    params = []

    # 定义条件模板
    conditions = {
        "location": (location, "location LIKE ?"),
        "name": (name, "name LIKE ?"),
    }

    # 处理普通条件
    for value, condition_template in conditions.values():
        if value:
            base_query += f" AND {condition_template}"
            params.append(f"%{value}%")  # 注意这里改为value而不是name

    # 处理特殊的关键词条件
    if keywords:
        keyword_list = [k.strip() for k in keywords.split(",")]
        keyword_conditions = " OR ".join(["keywords LIKE ?" for _ in keyword_list])
        base_query += f" AND ({keyword_conditions})"
        params.extend([f"%{keyword}%" for keyword in keyword_list])

    with sqlite3.connect(db) as conn:
        with conn.cursor() as cursor:
            cursor.execute(base_query, params)
            results = cursor.fetchall()

            conn.close()

            return [
                dict(zip([column[0] for column in cursor.description], row)) for row in results
            ]


@tool
def book_excursion(recommendation_id: int) -> str:
    """
    Book a excursion by its recommendation ID.

    Args:
        recommendation_id (int): The ID of the trip recommendation to book.

    Returns:
        str: A message indicating whether the trip recommendation was successfully booked or not.
    """
    with sqlite3.connect(db) as conn:
        with conn.cursor() as cursor:

            cursor.execute(
                "UPDATE trip_recommendations SET booked = 1 WHERE id = ?", (recommendation_id,)
            )
            conn.commit()

            if cursor.rowcount > 0:
                return f"Trip recommendation {recommendation_id} successfully booked."
            else:
                return f"No trip recommendation found with ID {recommendation_id}."


@tool
def update_excursion(recommendation_id: int, details: str) -> str:
    """
    Update a trip recommendation's details by its ID.

    Args:
        recommendation_id (int): The ID of the trip recommendation to update.
        details (str): The new details of the trip recommendation.

    Returns:
        str: A message indicating whether the trip recommendation was successfully updated or not.
    """
    with sqlite3.connect(db) as conn:
        with conn.cursor() as cursor:

            cursor.execute(
                "UPDATE trip_recommendations SET details = ? WHERE id = ?",
                (details, recommendation_id),
            )
            conn.commit()

            if cursor.rowcount > 0:
                return f"Trip recommendation {recommendation_id} successfully updated."
            else:
                return f"No trip recommendation found with ID {recommendation_id}."


@tool
def cancel_excursion(recommendation_id: int) -> str:
    """
    Cancel a trip recommendation by its ID.

    Args:
        recommendation_id (int): The ID of the trip recommendation to cancel.

    Returns:
        str: A message indicating whether the trip recommendation was successfully cancelled or not.
    """
    with sqlite3.connect(db) as conn:
        with conn.cursor() as cursor:

            cursor.execute(
                "UPDATE trip_recommendations SET booked = 0 WHERE id = ?", (recommendation_id,)
            )
            conn.commit()

            if cursor.rowcount > 0:
                return f"Trip recommendation {recommendation_id} successfully cancelled."
            else:
                return f"No trip recommendation found with ID {recommendation_id}."