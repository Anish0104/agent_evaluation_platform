import pytest
from pydantic import ValidationError

from app.tools.schemas import GetOrderArguments


def test_valid_order_id_is_accepted():
    arguments = GetOrderArguments.model_validate(
        {"order_id": "ord_1042"}
    )

    assert arguments.order_id == "ord_1042"


@pytest.mark.parametrize(
    "arguments",
    [
        {"order_id": 123},
        {"order_id": ""},
        {"order_id": "   "},
        {"order_id": "ord_1042", "admin": True},
        {},
    ],
)
def test_invalid_get_order_arguments_are_rejected(arguments):
    with pytest.raises(ValidationError):
        GetOrderArguments.model_validate(arguments)