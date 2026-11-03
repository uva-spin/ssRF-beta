from .model import Spin1Model, Spin1Params
from .lineshape import pake_component_raw, plot_signal_reference, boltzmann_branch_ratio
from .afp import (
    mirror_safe_indices,
    perform_afp_on_populations,
    qneg_sweep_indices,
    selective_window_indices,
)

__all__ = [
    "Spin1Model",
    "Spin1Params",
    "pake_component_raw",
    "plot_signal_reference",
    "boltzmann_branch_ratio",
    "mirror_safe_indices",
    "perform_afp_on_populations",
    "qneg_sweep_indices",
    "selective_window_indices",
]

# New ideal-control API; legacy names above remain available for regression.
from .ideal_model import IdealBinModel, IdealBinParams
from .pulse_program import BinPulse, RFProfile, PulseProgram, make_profile

__all__ += ["IdealBinModel", "IdealBinParams", "BinPulse", "RFProfile", "PulseProgram", "make_profile"]
