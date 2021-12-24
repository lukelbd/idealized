#!/usr/bin/env python3
"""
Shared package for working with idealized model experiments.
"""
import functools
import time
import warnings

_LoadWarning = type('LoadWarning', (UserWarning,), {})
_warn_simple = functools.partial(warnings.warn, category=_LoadWarning, stacklevel=2)


def _make_stopwatch(timer=False, fixedwidth=20):
    """
    Make a stopwatch using scary globals() fuckery.
    """
    if 't' not in globals():
        global t
        t = time.time()
    def _stopwatch(message):  # noqa: E306
        global t
        if timer:
            print(
                message + ':' + ' ' * (fixedwidth - len(message)),
                -t + (t := time.time())
            )
    return _stopwatch


from climopy import ureg, vreg, const  # noqa: F401
from . import definitions  # noqa: F401
from . import figures  # noqa: F401
from . import load  # noqa: F401
from . import videos  # noqa: F401
