from app.xai.base import BaseXAIMethod
from app.xai.gradcam_pp import GradCAMPlusPlusYOLO

_METHODS: dict[str, BaseXAIMethod] = {m.name: m for m in (GradCAMPlusPlusYOLO(),)}


def list_methods(architecture: str | None = None) -> list[BaseXAIMethod]:
    return [m for m in _METHODS.values()
            if architecture is None or architecture in m.applicable_architectures]


def get_method(name: str) -> BaseXAIMethod:
    return _METHODS[name]    # KeyError αν δεν υπάρχει