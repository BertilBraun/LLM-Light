from torch import nn

from llm_lite.config.models import InferenceConfiguration, Precision, QuantizationType


def prepare_model_for_inference(
    model: nn.Module,
    inference_configuration: InferenceConfiguration,
) -> nn.Module:
    _apply_precision(model=model, precision=inference_configuration.precision)
    _apply_quantization(model=model, quantization=inference_configuration.quantization)
    model.eval()
    return model


def _apply_precision(model: nn.Module, precision: Precision) -> None:
    match precision:
        case Precision.FP32:
            model.float()
        case Precision.FP16:
            model.half()
        case Precision.BF16:
            model.bfloat16()


def _apply_quantization(model: nn.Module, quantization: QuantizationType) -> None:
    match quantization:
        case QuantizationType.NONE:
            return
        case (
            QuantizationType.INT8_DYNAMIC
            | QuantizationType.INT8_WEIGHT_ONLY
            | QuantizationType.INT4_WEIGHT_ONLY
        ):
            raise ValueError(f"Quantization type {quantization.value!r} is not implemented.")
