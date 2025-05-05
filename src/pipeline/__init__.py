# This file makes the 'pipeline' directory a Python package.

# Optionally, import key components to make them accessible directly from the package
from .stage1_data_acquisition import NeuroReportDataModule
from .stage6_training import NeuroReportModel
from .stage10_main import main

# Or leave empty if you prefer explicit imports from submodules.
