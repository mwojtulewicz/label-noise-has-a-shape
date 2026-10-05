"""Geometry-guided exact-budget label-noise benchmark generators."""

from .baselines import (
    CIFAR10_SEMANTIC2_TARGETS,
    generate_semantic2_noise,
    generate_symmetric_noise,
    generate_xia_idln_noise,
)
from .generators import NoiseResult, generate_margin_noise, generate_proto_noise
from .geometry import Geometry, fit_geometry, true_class_margin
from .metrics import empirical_transition_matrix

__all__ = [
    "CIFAR10_SEMANTIC2_TARGETS",
    "Geometry",
    "NoiseResult",
    "empirical_transition_matrix",
    "fit_geometry",
    "generate_margin_noise",
    "generate_proto_noise",
    "generate_semantic2_noise",
    "generate_symmetric_noise",
    "generate_xia_idln_noise",
    "true_class_margin",
]
