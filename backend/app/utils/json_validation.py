from __future__ import annotations

from pydantic import BaseModel, ValidationError


def validate_payload(model_type: type[BaseModel], payload: dict, retries: int = 1) -> BaseModel:
    last_error: ValidationError | None = None
    attempts = retries + 1
    for _ in range(attempts):
        try:
            return model_type.model_validate(payload)
        except ValidationError as exc:
            last_error = exc
    if last_error is None:
        raise ValueError("validation failed without a concrete error")
    raise last_error

