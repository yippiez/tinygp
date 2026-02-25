from .ars import ARS
from .asebo import ASEBO
from .basic import BasicStrategy
from tinygp.definitions import StrategyState
from .cma_es import CMA_ES
from .cr_fm_nes import CR_FM_NES
from .differential_evolution import DifferentialEvolution
from .diffusion_evolution import DiffusionEvolution
from .discovered_es import DiscoveredES
from .esmc import ESMC
from .evotf_es import EvoTF_ES
from .gesmr_ga import GESMR_GA
from .gplearn_gp import GplearnGP
from .gradientless_descent import GradientlessDescent
from .guided_es import GuidedES
from .hill_climbing import HillClimbing
from .iamalgam_full import iAMaLGaM_Full
from .iamalgam_univariate import iAMaLGaM_Univariate
from .learned_es import LearnedES
from .learned_ga import LearnedGA
from .lm_ma_es import LM_MA_ES
from .ma_es import MA_ES
from .mr15_ga import MR15_GA
from .noise_reuse_es import NoiseReuseES
from .open_es import Open_ES
from .persistent_es import PersistentES
from .pgpe import PGPE
from .pso import PSO
from .random_search import RandomSearch
from .rm_es import Rm_ES
from .samr_ga import SAMR_GA
from .sep_cma_es import Sep_CMA_ES
from .sim_anneal import SimAnneal
from .simple_es import SimpleES
from .simple_ga import SimpleGA
from .snes import SNES
from .sv_cma_es import SV_CMA_ES
from .sv_openes import SV_OpenES
from .xnes import xNES


STRATEGY_REGISTRY = {
    "BasicStrategy": BasicStrategy,
    "SimpleES": SimpleES,
    "Open_ES": Open_ES,
    "CMA_ES": CMA_ES,
    "Sep_CMA_ES": Sep_CMA_ES,
    "xNES": xNES,
    "SNES": SNES,
    "MA_ES": MA_ES,
    "LM_MA_ES": LM_MA_ES,
    "Rm_ES": Rm_ES,
    "PGPE": PGPE,
    "ARS": ARS,
    "ESMC": ESMC,
    "PersistentES": PersistentES,
    "NoiseReuseES": NoiseReuseES,
    "CR_FM_NES": CR_FM_NES,
    "GuidedES": GuidedES,
    "ASEBO": ASEBO,
    "DiscoveredES": DiscoveredES,
    "LearnedES": LearnedES,
    "EvoTF_ES": EvoTF_ES,
    "iAMaLGaM_Full": iAMaLGaM_Full,
    "iAMaLGaM_Univariate": iAMaLGaM_Univariate,
    "GradientlessDescent": GradientlessDescent,
    "SimAnneal": SimAnneal,
    "HillClimbing": HillClimbing,
    "RandomSearch": RandomSearch,
    "SV_CMA_ES": SV_CMA_ES,
    "SV_OpenES": SV_OpenES,
    "SimpleGA": SimpleGA,
    "MR15_GA": MR15_GA,
    "SAMR_GA": SAMR_GA,
    "GESMR_GA": GESMR_GA,
    "GplearnGP": GplearnGP,
    "LearnedGA": LearnedGA,
    "DiffusionEvolution": DiffusionEvolution,
    "DifferentialEvolution": DifferentialEvolution,
    "PSO": PSO,
}


def create_strategy(name: str, *args, **kwargs):
    assert name in STRATEGY_REGISTRY, f"unknown strategy: {name}"
    return STRATEGY_REGISTRY[name](*args, **kwargs)


__all__ = [
    "BasicStrategy",
    "StrategyState",
    "SimpleES",
    "Open_ES",
    "CMA_ES",
    "Sep_CMA_ES",
    "xNES",
    "SNES",
    "MA_ES",
    "LM_MA_ES",
    "Rm_ES",
    "PGPE",
    "ARS",
    "ESMC",
    "PersistentES",
    "NoiseReuseES",
    "CR_FM_NES",
    "GuidedES",
    "ASEBO",
    "DiscoveredES",
    "LearnedES",
    "EvoTF_ES",
    "iAMaLGaM_Full",
    "iAMaLGaM_Univariate",
    "GradientlessDescent",
    "SimAnneal",
    "HillClimbing",
    "RandomSearch",
    "SV_CMA_ES",
    "SV_OpenES",
    "SimpleGA",
    "MR15_GA",
    "SAMR_GA",
    "GESMR_GA",
    "GplearnGP",
    "LearnedGA",
    "DiffusionEvolution",
    "DifferentialEvolution",
    "PSO",
    "STRATEGY_REGISTRY",
    "create_strategy",
]
