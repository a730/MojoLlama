import time


class RateLimiter:
    """A sliding window rate limiter that tracks requests per client.

    Limits the number of requests a client can make within a
    configurable time window.

    Args:
        max_requests: Maximum number of requests allowed within the window.
        window_seconds: Duration of the sliding window in seconds.
    """

    def __init__(self, max_requests, window_seconds):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}

    def allow_request(self, client_id):
        """Check whether a request from the given client should be allowed.

        Prunes expired timestamps for the client and, if the request
        count is below the limit, records the current time and returns
        ``True``.

        Args:
            client_id: Unique identifier for the client making the request.

        Returns:
            True if the request is allowed; False if the client has
            exceeded the rate limit.
        """
        now = time.time()
        if client_id not in self.requests:
            self.requests[client_id] = []
        self.requests[client_id] = [t for t in self.requests[client_id] if now - t < self.window_seconds]
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        self.requests[client_id].append(now)
        return True

    def get_remaining(self, client_id):
        """Return the number of requests remaining for the client.

        Args:
            client_id: Unique identifier for the client.

        Returns:
            The number of additional requests the client can make
            within the current window. Returns ``max_requests`` if
            the client has not yet made any requests.
        """
        now = time.time()
        if client_id not in self.requests:
            return self.max_requests
        self.requests[client_id] = [t for t in self.requests[client_id] if now - t < self.window_seconds]
        return max(0, self.max_requests - len(self.requests[client_id]))
