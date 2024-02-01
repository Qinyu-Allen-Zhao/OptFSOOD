from .ash import ASH
from .bfact import BFact
from .dice import DICE
from .ebo import EnergyBased
from .mls import MaxLogit
from .msp import MSP
from .odin import ODIN
from .react import React
from .surroact import SurrogateAct
from .vra import VRA_P
from .ours import Ours


def get_ood_detector(detector_type, benchmark_id, **kwargs):
    if detector_type == "msp":
        return MSP()
    elif detector_type == "ebo":
        return EnergyBased()
    elif detector_type == "mls":
        return MaxLogit()
    elif detector_type == "odin":
        return ODIN(noise=[0.004, 0.004, 0.0][benchmark_id])
    elif detector_type == "react":
        return React(kwargs['use_surrogate'])
    elif detector_type == "dice":
        return DICE([90, 90, 70][benchmark_id])
    elif detector_type == "vra":
        if benchmark_id <= 1:
            return VRA_P(0.6, 0.95, quantile=True, use_surrogate=kwargs['use_surrogate'])
        else:
            return VRA_P(use_surrogate=kwargs['use_surrogate'])
    elif detector_type == "ash_p":
        perc = [90, 80, 60][benchmark_id]
        return ASH(detector_type, perc, kwargs['use_surrogate'])
    elif detector_type == "ash_s":
        perc = [95, 90, 90][benchmark_id]
        return ASH(detector_type, perc, kwargs['use_surrogate'])
    elif detector_type == "ash_b":
        perc = [95, 85, 65][benchmark_id]
        return ASH(detector_type, perc, kwargs['use_surrogate'])
    elif detector_type == "bfact":
        return BFact(kwargs['use_surrogate'])
    elif detector_type == "ours":
        return Ours(use_ood_score=kwargs['use_ood_score'])
    elif detector_type == "surro":
        return SurrogateAct(kwargs['is_gaussian'], kwargs['use_real'])
    else:
        raise Exception()
