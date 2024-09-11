"""Template module file.

Functions should be specified in the public API and imported via root where appropriate.
Using google python coding style.
"""

import numpy as np
import pandas as pd
from beartype.typing import Any

# Public API
__all__ = ("foo",)


def foo(bar: int = 3) -> Any:
    """Test function.

    Seeing if all works and the dependencies have been installed from uv.

    Args:
        bar: some int input

    Returns:
        Any dataformat in this case testing numpy and pandas.
    """
    data = np.random.rand(bar, bar)
    df = pd.DataFrame({f"Column_{i}": data[:, i] for i in range(bar)})
    return df


if __name__ == "__main__":  # pragma: no cover
    print(foo())
