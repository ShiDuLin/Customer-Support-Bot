import sqlite3
from datetime import date, datetime
from typing import Optional, Union
from langchain_core.tools import tool
from db import db


@tool
def search_hotels(
    location: Optional[str] = None,
    name: Optional[str] = None,
    price_tier: Optional[str] = None,
    checkin_date: Optional[Union[datetime, date]] = None,
    checkout_date: Optional[Union[datetime, date]] = None,
) -> list[dict]:
    """
    Search for hotels based on location, name, price tier, check-in date, and check-out date.

    Args:
        location (Optional[str]): The location of the hotel. Defaults to None.
        name (Optional[str]): The name of the hotel. Defaults to None.
        price_tier (Optional[str]): The price tier of the hotel. Defaults to None. Examples: Midscale, Upper Midscale, Upscale, Luxury
        checkin_date (Optional[Union[datetime, date]]): The check-in date of the hotel. Defaults to None.
        checkout_date (Optional[Union[datetime, date]]): The check-out date of the hotel. Defaults to None.

    Returns:
        list[dict]: A list of hotel dictionaries matching the search criteria.
    """
    base_query = "SELECT * FROM hotels WHERE 1=1"
    params = []

    conditions = {
        "location": (location, "location LIKE ?"),
        "name": (name, "name LIKE ?"),
    }
    
    for value, condition in conditions.values():
        if value:
            base_query += f" AND {condition}"
            params.append(f"%{value}%")
    
    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()

        # For the sake of this tutorial, we will let you match on any dates and price tier.
        cursor.execute(base_query, params)
        results = [
            dict(zip([column[0] for column in cursor.description], row)) for row in cursor.fetchall()
        ]

        cursor.close()
        return results


@tool
def book_hotel(hotel_id: int) -> str:
    """
    Book a hotel by its ID.

    Args:
        hotel_id (int): The ID of the hotel to book.

    Returns:
        str: A message indicating whether the hotel was successfully booked or not.
    """
    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()

        cursor.execute("UPDATE hotels SET booked = 1 WHERE id = ?", (hotel_id,))
        conn.commit()

        if cursor.rowcount > 0:
            cursor.close()
            return f"Hotel {hotel_id} successfully booked."
        else:
            cursor.close()
            return f"No hotel found with ID {hotel_id}."


@tool
def update_hotel(
    hotel_id: int,
    checkin_date: Optional[Union[datetime, date]] = None,
    checkout_date: Optional[Union[datetime, date]] = None,
) -> str:
    """
    Update a hotel's check-in and check-out dates by its ID.

    Args:
        hotel_id (int): The ID of the hotel to update.
        checkin_date (Optional[Union[datetime, date]]): The new check-in date of the hotel. Defaults to None.
        checkout_date (Optional[Union[datetime, date]]): The new check-out date of the hotel. Defaults to None.

    Returns:
        str: A message indicating whether the hotel was successfully updated or not.
    """
    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()

        if checkin_date:
            cursor.execute(
                "UPDATE hotels SET checkin_date = ? WHERE id = ?", (checkin_date, hotel_id)
            )
        if checkout_date:
            cursor.execute(
                "UPDATE hotels SET checkout_date = ? WHERE id = ?",
                (checkout_date, hotel_id),
            )

        conn.commit()

        if cursor.rowcount > 0:
            cursor.close()
            return f"Hotel {hotel_id} successfully updated."
        else:
            cursor.close()
            return f"No hotel found with ID {hotel_id}."


@tool
def cancel_hotel(hotel_id: int) -> str:
    """
    Cancel a hotel by its ID.

    Args:
        hotel_id (int): The ID of the hotel to cancel.

    Returns:
        str: A message indicating whether the hotel was successfully cancelled or not.
    """
    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()

        cursor.execute("UPDATE hotels SET booked = 0 WHERE id = ?", (hotel_id,))
        conn.commit()

        if cursor.rowcount > 0:
            cursor.close()
            return f"Hotel {hotel_id} successfully cancelled."
        else:
            cursor.close()
            return f"No hotel found with ID {hotel_id}."