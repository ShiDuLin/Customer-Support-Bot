import sqlite3
from datetime import date, datetime
from typing import Optional, Union
from langchain_core.tools import tool
from db.db import db


@tool
def search_car_rentals(
    location: Optional[str] = None,
    name: Optional[str] = None,
    price_tier: Optional[str] = None,
    start_date: Optional[Union[datetime, date]] = None,
    end_date: Optional[Union[datetime, date]] = None,
) -> list[dict]:
    """
    Search for car rentals based on location, name, price tier, start date, and end date.

    Args:
        location (Optional[str]): The location of the car rental. Defaults to None.
        name (Optional[str]): The name of the car rental company. Defaults to None.
        price_tier (Optional[str]): The price tier of the car rental. Defaults to None.
        start_date (Optional[Union[datetime, date]]): The start date of the car rental. Defaults to None.
        end_date (Optional[Union[datetime, date]]): The end date of the car rental. Defaults to None.

    Returns:
        list[dict]: A list of car rental dictionaries matching the search criteria.
    """
    base_query = "SELECT * FROM car_rentals WHERE 1=1"
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
        # For our tutorial, we will let you match on any dates and price tier.
        # (since our toy dataset doesn't have much data)
        cursor.execute(base_query, params)
        results = [
            dict(zip([column[0] for column in cursor.description], row)) for row in cursor.fetchall()
        ]
        cursor.close()
        return results


@tool
def book_car_rental(rental_id: int) -> str:
    """
    Book a car rental by its ID.

    Args:
        rental_id (int): The ID of the car rental to book.

    Returns:
        str: A message indicating whether the car rental was successfully booked or not.
    """
    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()

        cursor.execute("UPDATE car_rentals SET booked = 1 WHERE id = ?", (rental_id,))
        conn.commit()

        if cursor.rowcount > 0:
            cursor.close()
            return f"Car rental {rental_id} successfully booked."
        else:
            cursor.close()
            return f"No car rental found with ID {rental_id}."


@tool
def update_car_rental(
    rental_id: int,
    start_date: Optional[Union[datetime, date]] = None,
    end_date: Optional[Union[datetime, date]] = None,
) -> str:
    """
    Update a car rental's start and end dates by its ID.

    Args:
        rental_id (int): The ID of the car rental to update.
        start_date (Optional[Union[datetime, date]]): The new start date of the car rental. Defaults to None.
        end_date (Optional[Union[datetime, date]]): The new end date of the car rental. Defaults to None.

    Returns:
        str: A message indicating whether the car rental was successfully updated or not.
    """
    # base_query = "SELECT * FROM car_rentals WHERE 1=1"
    # params = []

    # conditions = {
    #     "location": (location, "location LIKE ?"),
    #     "name": (name, "name LIKE ?"),
    # }
    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()

        if start_date:
            cursor.execute(
                "UPDATE car_rentals SET start_date = ? WHERE id = ?",
                (start_date, rental_id),
            )
        if end_date:
            cursor.execute(
                "UPDATE car_rentals SET end_date = ? WHERE id = ?", (end_date, rental_id)
            )

        conn.commit()

        if cursor.rowcount > 0:
            cursor.close()
            return f"Car rental {rental_id} successfully updated."
        else:
            cursor.close()
            return f"No car rental found with ID {rental_id}."


@tool
def cancel_car_rental(rental_id: int) -> str:
    """
    Cancel a car rental by its ID.

    Args:
        rental_id (int): The ID of the car rental to cancel.

    Returns:
        str: A message indicating whether the car rental was successfully cancelled or not.
    """
    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()

        cursor.execute("UPDATE car_rentals SET booked = 0 WHERE id = ?", (rental_id,))
        conn.commit()

        if cursor.rowcount > 0:
            cursor.close()
            return f"Car rental {rental_id} successfully cancelled."
        else:
            cursor.close()
            return f"No car rental found with ID {rental_id}."