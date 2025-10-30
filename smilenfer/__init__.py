import warnings

warnings.resetwarnings()  # wipe whatever filters are set
# Ignore the usual offenders everywhere, not just your code
_ignore = [
    DeprecationWarning,
    FutureWarning,
    PendingDeprecationWarning,
    UserWarning,
    RuntimeWarning,
]
for _cat in _ignore:
    warnings.filterwarnings("ignore", category=_cat)

# Numpy’s special warning class
try:
    import numpy as _np
    warnings.filterwarnings("ignore", category=_np.VisibleDeprecationWarning)
except Exception:
    pass

del _ignore, _cat, _np
