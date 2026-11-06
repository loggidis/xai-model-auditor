from app.xai.base import BaseXAIMethod
from app.xai.gradcam_pp import GradCAMPlusPlusYOLO
from app.xai.integrated_gradients import IntegratedGradientsYOLO
from app.xai.lime_image import LimeImageYOLO
from app.xai.shap_partition import ShapPartitionYOLO

_METHODS: dict[str, BaseXAIMethod] = {
    m.name: m for m in (GradCAMPlusPlusYOLO(), IntegratedGradientsYOLO(),
                        ShapPartitionYOLO(), LimeImageYOLO())
}


def list_methods(architecture: str | None = None) -> list[BaseXAIMethod]:
    return [m for m in _METHODS.values()
            if architecture is None or architecture in m.applicable_architectures]


def get_method(name: str) -> BaseXAIMethod:
    return _METHODS[name]    # KeyError if it does not exist