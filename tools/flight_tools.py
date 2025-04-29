import sqlite3
from datetime import date, datetime
from typing import Union

import pytz
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from db.db import db
# from db.retriever import lookup_policy

@tool
def fetch_user_flight_information(config: RunnableConfig) -> list[dict]:
    """Fetch all tickets for the user along with corresponding flight information and seat assignments.

    Returns:
        A list of dictionaries where each dictionary contains the ticket details,
        associated flight details, and the seat assignments for each ticket belonging to the user.
    """
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                t.ticket_no, t.book_ref,
                f.flight_id, f.flight_no, f.departure_airport, f.arrival_airport,
                f.scheduled_departure, f.scheduled_arrival,
                bp.seat_no, tf.fare_conditions
            FROM tickets t
            JOIN ticket_flights tf ON t.ticket_no = tf.ticket_no
            JOIN flights f ON tf.flight_id = f.flight_id
            JOIN boarding_passes bp ON bp.ticket_no = t.ticket_no 
                AND bp.flight_id = f.flight_id
            WHERE t.passenger_id = ?
            LIMIT 100  -- 防爆措施
        """, (passenger_id,))
        result = [
            dict(zip(
                [col[0] for col in cursor.description], 
                row
            ))
            for row in cursor.fetchall()
        ]
        cursor.close()
        return result


@tool
def search_flights(
    departure_airport: Union[str, None],
    arrival_airport: Union[str, None],
    start_time: Union[date | datetime, None],
    end_time: Union[date | datetime, None],
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    """Search for flights based on departure airport, arrival airport, and departure time range."""
    base_query = """
    SELECT 
        flight_id, flight_no, 
        departure_airport, arrival_airport,
        scheduled_departure, scheduled_arrival
    FROM flights 
    WHERE 1=1
    """
    params = []
    
    # 动态构建查询
    conditions = {
        "departure": (departure_airport, "departure_airport = ?"),
        "arrival": (arrival_airport, "arrival_airport = ?"),
        "start": (start_time, "scheduled_departure >= ?"),
        "end": (end_time, "scheduled_departure <= ?")
    }
    
    for value, condition in conditions.values():
        if value:
            base_query += f" AND {condition}"
            params.append(value)
    
    # 分页控制
    base_query += " LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    # 执行查询
    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()
        cursor.execute(base_query, params)
        result = [
            dict(zip(
                [col[0] for col in cursor.description], 
                row
            ))
            for row in cursor.fetchall()
        ]
        cursor.close()
        return result
        

@tool(return_direct=True)
def update_ticket_to_new_flight(
    ticket_no: str, 
    new_flight_id: int, 
    *, 
    config: RunnableConfig
) -> str:
    """Updates a ticket to a new flight after validating ownership, flight validity, and timing rules."""
    
    # 身份验证
    if not (passenger_id := config.get("configurable", {}).get("passenger_id")):
        raise ValueError("Passenger ID required in config.configurable")

    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()
        # 验证目标航班
        cursor.execute(
            """SELECT departure_airport, arrival_airport, scheduled_departure 
            FROM flights WHERE flight_id = ?""",
            (new_flight_id,)
        )
        if not (new_flight := cursor.fetchone()):
            return "Invalid new flight ID"
            
        # 解析航班时间
        dep_airport, arr_airport, dep_time_str = new_flight
        dep_time = datetime.fromisoformat(dep_time_str)
        if (dep_time - datetime.now(pytz.utc)).total_seconds() < 10800:
            cursor.close()
            return f"Cannot reschedule to flight departing in <3 hours ({dep_time})"

        # 验证机票所有权
        cursor.execute(
            """SELECT 1 FROM tickets t
            JOIN ticket_flights tf ON t.ticket_no = tf.ticket_no
            WHERE t.ticket_no = ? AND t.passenger_id = ?""",
            (ticket_no, passenger_id)
        )
        if not cursor.fetchone():
            cursor.close()
            return f"Passenger {passenger_id} does not own ticket {ticket_no}"

        # 执行改签
        cursor.execute(
            "UPDATE ticket_flights SET flight_id = ? WHERE ticket_no = ?",
            (new_flight_id, ticket_no))
        conn.commit()
        cursor.close()
            
    return f"Successfully updated ticket {ticket_no} to flight {new_flight_id}"


@tool
def cancel_ticket(ticket_no: str, *, config: RunnableConfig) -> str:
    """Cancel the user's ticket and remove it from the database."""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")
    
    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (ticket_no,)
        )
    
        if not cursor.fetchone():
            cursor.close()
            return "No existing ticket found for the given ticket number."

        # Check the signed-in user actually has this ticket
        cursor.execute(
            "SELECT ticket_no FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
            (ticket_no, passenger_id),
        )
        
        if not cursor.fetchone():
            cursor.close()
            return f"Current signed-in passenger with ID {passenger_id} not the owner of ticket {ticket_no}"

        cursor.execute("DELETE FROM ticket_flights WHERE ticket_no = ?", (ticket_no,))
        conn.commit()
        cursor.close()

    return "Ticket successfully cancelled."