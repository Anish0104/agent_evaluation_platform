from pydantic import BaseModel, ConfigDict, Field, field_validator


class GetOrderArguments(BaseModel):
    """Accepted agent-provided arguments for get_order."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
    )

    order_id: str = Field(min_length=1)

    @field_validator("order_id")
    @classmethod
    def reject_blank_order_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("order_id must not be blank.")
        return value