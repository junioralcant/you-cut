from __future__ import annotations

from collections.abc import Callable
from contextlib import ContextDecorator
from dataclasses import dataclass
from functools import wraps
from typing import Any

import httpx


@dataclass
class _Route:
    method: str
    url: str
    side_effect: Any = None
    return_value: httpx.Response | None = None
    call_count: int = 0

    def mock(
        self,
        *,
        side_effect: Any = None,
        return_value: httpx.Response | None = None,
    ) -> "_Route":
        self.side_effect = side_effect
        self.return_value = return_value
        return self


_ROUTES: list[_Route] = []


def post(url: str) -> _Route:
    route = _Route(method="POST", url=url)
    _ROUTES.append(route)
    return route


def _response_with_request(response: httpx.Response, request: httpx.Request) -> httpx.Response:
    try:
        if response.request is not None:
            return response
    except RuntimeError:
        pass
    return httpx.Response(
        response.status_code,
        headers=response.headers,
        content=response.content,
        request=request,
        extensions=response.extensions,
    )


def _dispatch(method: str, url: str, **kwargs: Any) -> httpx.Response:
    request = httpx.Request(method, url, json=kwargs.get("json"), data=kwargs.get("data"))

    for route in _ROUTES:
        if route.method == method and route.url == url:
            route.call_count += 1
            effect = route.side_effect
            if isinstance(effect, list):
                if not effect:
                    raise AssertionError(f"No more mocked responses for {method} {url}")
                outcome = effect.pop(0)
            elif callable(effect):
                outcome = effect(request)
            elif effect is not None:
                outcome = effect
            elif route.return_value is not None:
                outcome = route.return_value
            else:
                outcome = httpx.Response(200, request=request)

            if isinstance(outcome, Exception):
                raise outcome
            if not isinstance(outcome, httpx.Response):
                raise TypeError(f"Unsupported mocked outcome for {method} {url}: {type(outcome)!r}")
            return _response_with_request(outcome, request)

    raise AssertionError(f"Unmocked request: {method} {url}")


class _MockRouter(ContextDecorator):
    def __enter__(self) -> "_MockRouter":
        self._orig_post = httpx.post
        _ROUTES.clear()

        def _mocked_post(url: str, **kwargs: Any) -> httpx.Response:
            return _dispatch("POST", url, **kwargs)

        httpx.post = _mocked_post
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        httpx.post = self._orig_post
        _ROUTES.clear()
        return False

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self.__class__():
                return func(*args, **kwargs)

        return wrapper


mock = _MockRouter()
