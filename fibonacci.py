def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number using iterative approach.

    Args:
        n: The position in the Fibonacci sequence (0-indexed).

    Returns:
        The nth Fibonacci number where fibonacci(0) = 0, fibonacci(1) = 1.

    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
