"""Repository scripts, importable so their logic can be tested.

The metric-regression gate lives in ``check_metric_regression.py`` and is exercised by
``tests/tripath/test_metric_regression_gate.py``; a gate with no test cannot be trusted
to fail. This package is not part of the distribution - ``pyproject.toml`` ships
``docureason*`` only.
"""
