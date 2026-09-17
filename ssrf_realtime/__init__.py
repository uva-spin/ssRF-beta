from .model import Spin1Model, Spin1Params
from .lineshape import pake_component_raw, plot_signal_reference, boltzmann_branch_ratio

__all__ = [
    "Spin1Model",
    "Spin1Params",
    "pake_component_raw",
    "plot_signal_reference",
    "boltzmann_branch_ratio",
]

# New ideal-control API; legacy names above remain available for regression.
from .ideal_model import IdealBinModel, IdealBinParams
from .pulse_program import BinPulse, RFProfile, PulseProgram, make_profile

__all__ += ["IdealBinModel", "IdealBinParams", "BinPulse", "RFProfile", "PulseProgram", "make_profile"]
