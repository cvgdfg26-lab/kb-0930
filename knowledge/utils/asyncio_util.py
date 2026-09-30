import asyncio
import errno
import socket
import sys
from typing import Any, Dict


def configure_windows_event_loop_policy() -> None:
    """Prefer the selector loop on Windows to reduce noisy socket close callbacks."""
    if sys.platform != "win32":
        return

    policy_cls = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if policy_cls is not None:
        asyncio.set_event_loop_policy(policy_cls())


def install_asyncio_connection_reset_filter(loop: asyncio.AbstractEventLoop) -> None:
    """Silence benign Windows socket-reset errors during transport shutdown.

    FastAPI/SSE clients and MCP SSE connections may close their TCP connection while
    Python is already tearing down the transport. On Windows this can surface as an
    unhandled callback error from asyncio's Proactor transport, even though the
    request has completed normally.
    """
    previous_handler = loop.get_exception_handler()

    def handler(active_loop: asyncio.AbstractEventLoop, context: Dict[str, Any]) -> None:
        exception = context.get("exception")
        if _is_benign_connection_reset(exception):
            return

        if previous_handler is not None:
            previous_handler(active_loop, context)
            return

        active_loop.default_exception_handler(context)

    loop.set_exception_handler(handler)


def _is_benign_connection_reset(exception: BaseException | None) -> bool:
    if isinstance(exception, (ConnectionResetError, BrokenPipeError)):
        return True

    if isinstance(exception, OSError):
        winerror = getattr(exception, "winerror", None)
        return winerror == 10054 or exception.errno in {
            errno.ECONNRESET,
            errno.EPIPE,
            errno.ENOTCONN,
            socket.WSAECONNRESET if hasattr(socket, "WSAECONNRESET") else 10054,
        }

    return False
