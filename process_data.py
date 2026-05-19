from typing import Any


class InvalidItemTypeError(ValueError):
    """Raised when an item has an invalid or missing 'type' field."""


class MissingFieldError(KeyError):
    """Raised when a required field is missing from an item."""


def _process_user_item(item: dict[str, Any]) -> dict[str, Any]:
    if "id" not in item or "name" not in item:
        raise MissingFieldError("User items must have 'id' and 'name' fields")
    return {"id": item["id"], "name": item["name"], "role": "user"}


def _process_admin_item(item: dict[str, Any]) -> dict[str, Any]:
    if "id" not in item or "name" not in item:
        raise MissingFieldError("Admin items must have 'id' and 'name' fields")
    return {
        "id": item["id"],
        "name": item["name"],
        "role": "admin",
        "permissions": ["read", "write", "delete"],
    }


def _process_guest_item(item: dict[str, Any]) -> dict[str, Any]:
    if "id" not in item:
        raise MissingFieldError("Guest items must have an 'id' field")
    return {
        "id": item["id"],
        "name": item.get("name", "Guest"),
        "role": "guest",
    }


_ITEM_PROCESSORS: dict[str, Any] = {
    "user": _process_user_item,
    "admin": _process_admin_item,
    "guest": _process_guest_item,
}


def process_item(item: dict[str, Any]) -> dict[str, Any]:
    item_type = item.get("type")
    if item_type not in _ITEM_PROCESSORS:
        raise InvalidItemTypeError(f"Unknown or missing item type: {item_type!r}")
    return _ITEM_PROCESSORS[item_type](item)


def process_data(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = [process_item(item) for item in data]
    result.sort(key=lambda x: x["id"])
    return result
