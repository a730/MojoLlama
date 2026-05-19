import unittest

from process_data import (
    InvalidItemTypeError,
    MissingFieldError,
    process_data,
    process_item,
)


class TestProcessItem(unittest.TestCase):
    def test_user_item(self):
        item = {"type": "user", "id": 1, "name": "Alice"}
        result = process_item(item)
        self.assertEqual(result, {"id": 1, "name": "Alice", "role": "user"})

    def test_admin_item(self):
        item = {"type": "admin", "id": 2, "name": "Bob"}
        result = process_item(item)
        self.assertEqual(
            result,
            {
                "id": 2,
                "name": "Bob",
                "role": "admin",
                "permissions": ["read", "write", "delete"],
            },
        )

    def test_guest_item_with_name(self):
        item = {"type": "guest", "id": 3, "name": "Charlie"}
        result = process_item(item)
        self.assertEqual(result, {"id": 3, "name": "Charlie", "role": "guest"})

    def test_guest_item_without_name(self):
        item = {"type": "guest", "id": 4}
        result = process_item(item)
        self.assertEqual(result, {"id": 4, "name": "Guest", "role": "guest"})

    def test_missing_type_raises_error(self):
        with self.assertRaises(InvalidItemTypeError):
            process_item({"id": 1, "name": "Alice"})

    def test_unknown_type_raises_error(self):
        with self.assertRaises(InvalidItemTypeError):
            process_item({"type": "unknown", "id": 1, "name": "Alice"})

    def test_user_missing_id_raises_error(self):
        with self.assertRaises(MissingFieldError):
            process_item({"type": "user", "name": "Alice"})

    def test_user_missing_name_raises_error(self):
        with self.assertRaises(MissingFieldError):
            process_item({"type": "user", "id": 1})

    def test_admin_missing_id_raises_error(self):
        with self.assertRaises(MissingFieldError):
            process_item({"type": "admin", "name": "Bob"})

    def test_guest_missing_id_raises_error(self):
        with self.assertRaises(MissingFieldError):
            process_item({"type": "guest"})


class TestProcessData(unittest.TestCase):
    def test_mixed_items(self):
        data = [
            {"type": "user", "id": 3, "name": "Charlie"},
            {"type": "admin", "id": 1, "name": "Alice"},
            {"type": "guest", "id": 2},
        ]
        result = process_data(data)
        self.assertEqual(
            result,
            [
                {
                    "id": 1,
                    "name": "Alice",
                    "role": "admin",
                    "permissions": ["read", "write", "delete"],
                },
                {"id": 2, "name": "Guest", "role": "guest"},
                {"id": 3, "name": "Charlie", "role": "user"},
            ],
        )

    def test_empty_list(self):
        self.assertEqual(process_data([]), [])

    def test_sorts_by_id(self):
        data = [
            {"type": "user", "id": 10, "name": "Zara"},
            {"type": "user", "id": 1, "name": "Amy"},
        ]
        result = process_data(data)
        self.assertEqual(result[0]["id"], 1)
        self.assertEqual(result[1]["id"], 10)

    def test_invalid_item_raises_error(self):
        data = [
            {"type": "user", "id": 1, "name": "Alice"},
            {"type": "invalid"},
        ]
        with self.assertRaises(InvalidItemTypeError):
            process_data(data)


if __name__ == "__main__":
    unittest.main()
