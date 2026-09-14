"""Keep terminal I/O failures separate from training failures."""

from __future__ import annotations

import builtins
import sys
from contextlib import contextmanager


class ResilientTextStream:
    """Drop failed output, retry on the next write, and report recovery."""

    def __init__(self, stream):
        self.stream = stream
        self.failed_operations = 0
        self.pending_failures = 0
        self.last_error = None

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def _record_failure(self, error):
        self.failed_operations += 1
        self.pending_failures += 1
        self.last_error = str(error)

    def write(self, text):
        try:
            if self.pending_failures:
                self.stream.write(
                    f"\nWARNING: console recovered after {self.pending_failures} "
                    f"failed output operations ({self.last_error}); training continued.\n"
                )
                self.stream.flush()
                self.pending_failures = 0
            return self.stream.write(text)
        except OSError as error:
            self._record_failure(error)
            return len(text)

    def flush(self):
        try:
            self.stream.flush()
        except OSError as error:
            self._record_failure(error)

    def isatty(self):
        try:
            return self.stream.isatty()
        except OSError:
            return False


def safe_print(*args, **kwargs):
    """Used also when a trainer is called without the CLI console guard."""
    try:
        builtins.print(*args, **kwargs)
    except OSError:
        # A broken warning sink must not abort checkpoint retries either.
        pass


@contextmanager
def resilient_console():
    """Cover print, tqdm, Colorama and console capture for one training run."""
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout = ResilientTextStream(stdout)
    sys.stderr = ResilientTextStream(stderr)
    try:
        yield
    finally:
        # Do not let shutdown flushing hide the original training exception.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except OSError:
                pass
        sys.stdout, sys.stderr = stdout, stderr
