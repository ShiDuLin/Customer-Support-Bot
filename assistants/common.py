from langgraph.graph import MessagesState
from typing import Annotated, Literal, Union


def update_dialog_stack(left: list[str], right: Union[str, None]) -> list[str]:
    """Push or pop the state."""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]


class State(MessagesState):
    user_info: str
    dialog_state: Annotated[
        list[
            Literal[
                "primary_assistant",
                "update_flight",
                "book_car_rental",
                "book_hotel",
                "book_excursion",
            ]
        ],
        update_dialog_stack,
    ]
