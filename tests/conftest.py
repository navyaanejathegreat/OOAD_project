import os
import sys
from datetime import datetime

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

T0 = datetime(2026, 7, 15, 17, 0)


@pytest.fixture
def t0():
    return T0
