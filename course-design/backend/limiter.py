# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared rate limiter for the application.

Provides a global SlowAPI rate limiter instance that limits API requests
to 60 per minute per client IP address by default. Individual routes can
override the default limit via the ``@app_limiter.limit()`` decorator.

Attributes:
    app_limiter: SlowAPI Limiter instance bound to client IP addresses.
"""

# Try to import slowapi, otherwise provide a dummy limiter
try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    app_limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
except ImportError:
    # Dummy limiter decorator that does nothing
    def dummy_limiter(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    class DummyLimiter:
        def limit(self, *args, **kwargs):
            return dummy_limiter
    
    app_limiter = DummyLimiter()
