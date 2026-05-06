try:
    from ._discodj_native import *
    from ._discodj_native import __doc__
    _NATIVE_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    _NATIVE_IMPORT_ERROR = exc
    __doc__ = (
        "Fallback Python shim for discodj_native. The compiled "
        "_discodj_native extension is not available in this checkout."
    )
    __all__ = ["nbody2d", "nbody3d", "rng_ngenic"]

    def _raise_missing_native(*args, **kwargs):
        raise ModuleNotFoundError(
            "The compiled 'discodj_native._discodj_native' extension is not "
            "available. Build/install the native DISCO-DJ extension before "
            "using TreePM exact short-range forces or N-GenIC-compatible RNGs."
        ) from _NATIVE_IMPORT_ERROR

    def nbody2d(*args, **kwargs):
        return _raise_missing_native(*args, **kwargs)

    def nbody3d(*args, **kwargs):
        return _raise_missing_native(*args, **kwargs)

    def rng_ngenic(*args, **kwargs):
        return _raise_missing_native(*args, **kwargs)

import sys


# set docstrings of this python module of the _discodj_native pybind11 module
_module = sys.modules[__name__]
for name in dir(_module):
    obj = getattr(_module, name)
    try:
        obj.__module__ = __name__
    except (AttributeError, TypeError):
        # not all objects allow __module__ to be set
        pass
