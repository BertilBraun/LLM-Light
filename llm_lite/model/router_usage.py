from torch import nn

from llm_lite.model.moe import MoeGpt
from llm_lite.model.routing import RouterUsageSummary


def collect_router_usage_summaries(model: nn.Module) -> tuple[RouterUsageSummary, ...]:
    match model:
        case MoeGpt():
            return model.router_usage_summaries()
        case _:
            return ()


def reset_router_usage(model: nn.Module) -> None:
    match model:
        case MoeGpt():
            model.reset_router_usage()
        case _:
            return
