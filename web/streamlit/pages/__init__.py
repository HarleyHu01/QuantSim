"""Streamlit 页面。

每个页面模块暴露 ``render(api_client)``，由 ``streamlit_app.py`` 按导航调用。
"""
from . import ai_page  # noqa: F401
from . import backtest_page  # noqa: F401
from . import data_page  # noqa: F401
from . import data_settings_page  # noqa: F401
from . import fusion_page  # noqa: F401
from . import kronos_page  # noqa: F401
from . import paper_page  # noqa: F401
from . import portfolio_page  # noqa: F401
from . import settings_page  # noqa: F401
from . import strategies_page  # noqa: F401
from . import strategy_create_page  # noqa: F401
from . import strategy_settings_page  # noqa: F401
