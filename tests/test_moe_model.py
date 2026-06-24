import json
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

from llm_lite.config.models import (
    DenseGptConfiguration,
    ModelType,
    MoeGptConfiguration,
)
from llm_lite.model.factory import build_model
from llm_lite.model.gpt import DenseGpt
from llm_lite.model.moe import MoeGpt
from llm_lite.model.output import ModelOutput
from llm_lite.model.parameters import model_parameter_summary
from llm_lite.model.routing import TopKRouter
from llm_lite.pipeline.runner import run_pipeline
from llm_lite.training.checkpoint import load_latest_checkpoint, save_checkpoint
from llm_lite.training.objectives import (
    CausalLanguageModelingObjectiveRunner,
    causal_language_modeling_loss,
)


def test_model_factory_returns_dense_and_moe_models() -> None:
    dense_model = build_model(
        model_configuration=_dense_configuration(),
        vocabulary_size=16,
    )
    moe_model = build_model(
        model_configuration=_moe_configuration(),
        vocabulary_size=16,
    )

    assert isinstance(dense_model, DenseGpt)
    assert isinstance(moe_model, MoeGpt)


def test_moe_forward_output_shape_and_auxiliary_loss() -> None:
    torch.manual_seed(3)
    model = MoeGpt(model_configuration=_moe_configuration(), vocabulary_size=19)
    token_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)

    model_output = model(token_ids)

    assert model_output.logits.shape == (2, 4, 19)
    assert model_output.auxiliary_loss is not None
    assert torch.isfinite(model_output.auxiliary_loss)


def test_moe_parameter_summary_reports_active_parameters() -> None:
    model = MoeGpt(model_configuration=_moe_configuration(), vocabulary_size=19)

    parameter_summary = model_parameter_summary(model=model)

    assert parameter_summary.total_parameters > parameter_summary.active_parameters
    assert parameter_summary.trainable_parameters > parameter_summary.trainable_active_parameters


def test_router_top_k_shape_and_deterministic_routing() -> None:
    router = TopKRouter(dimension=3, expert_count=4, top_k=2)
    with torch.no_grad():
        router.projection.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [-1.0, -1.0, -1.0],
                ],
            ),
        )
    hidden_states = torch.tensor(
        [[[5.0, 1.0, 0.0], [0.0, 4.0, 2.0]]],
        dtype=torch.float32,
    )

    routing_result = router(hidden_states=hidden_states)

    assert routing_result.router_logits.shape == (1, 2, 4)
    assert routing_result.top_expert_indices.shape == (1, 2, 2)
    assert routing_result.top_expert_weights.shape == (1, 2, 2)
    assert routing_result.top_expert_indices.tolist() == [[[0, 1], [1, 2]]]


def test_causal_lm_objective_includes_auxiliary_loss_when_configured() -> None:
    token_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    logits = torch.zeros((1, 3, 4), dtype=torch.float32)
    auxiliary_loss = torch.tensor(2.5)
    model = _FixedOutputModel(
        model_output=ModelOutput(logits=logits, auxiliary_loss=auxiliary_loss),
    )
    runner = CausalLanguageModelingObjectiveRunner(auxiliary_loss_weight=0.2)

    loss = runner.loss(model=model, batch=token_ids)

    expected_loss = causal_language_modeling_loss(logits=logits, token_ids=token_ids) + 0.5
    assert torch.allclose(loss, expected_loss)


def test_moe_checkpoint_save_and_load_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(11)
    model = MoeGpt(model_configuration=_moe_configuration(), vocabulary_size=17)
    optimizer = AdamW(model.parameters(), lr=0.01)
    checkpoint_directory = tmp_path / "checkpoints"
    save_checkpoint(
        checkpoint_directory=checkpoint_directory,
        model=model,
        optimizer=optimizer,
        step=4,
    )
    loaded_model = MoeGpt(model_configuration=_moe_configuration(), vocabulary_size=17)
    loaded_optimizer = AdamW(loaded_model.parameters(), lr=0.01)

    loaded_step = load_latest_checkpoint(
        checkpoint_directory=checkpoint_directory,
        model=loaded_model,
        optimizer=loaded_optimizer,
    )

    assert loaded_step == 4
    for parameter, loaded_parameter in zip(
        model.parameters(),
        loaded_model.parameters(),
        strict=True,
    ):
        assert torch.equal(parameter, loaded_parameter)


def test_tiny_pipeline_config_trains_moe_for_a_few_steps(tmp_path: Path) -> None:
    run_directory = tmp_path / "moe_smoke"
    configuration_path = tmp_path / "moe_smoke.yaml"
    configuration_text = Path("tests/configs/moe_smoke.yaml").read_text(encoding="utf-8")
    configuration_path.write_text(
        configuration_text.replace(
            "output_dir: runs/moe_smoke",
            f"output_dir: {str(run_directory).replace(chr(92), '/')}",
        ),
        encoding="utf-8",
    )

    exit_code = run_pipeline(
        configuration_path=configuration_path,
        dry_run=False,
        force_stages=(),
    )
    pretraining_manifest = json.loads(
        (run_directory / "artifacts" / "pretraining" / "manifest.json").read_text(
            encoding="utf-8",
        ),
    )

    assert exit_code == 0
    assert pretraining_manifest["metrics"]["final_step"] == 50
    assert (
        pretraining_manifest["metrics"]["model_parameters"]
        > pretraining_manifest["metrics"]["active_model_parameters"]
    )
    assert (run_directory / "artifacts" / "pretraining" / "checkpoints" / "latest.pt").exists()


def _dense_configuration() -> DenseGptConfiguration:
    return DenseGptConfiguration(
        type=ModelType.DENSE_GPT,
        dimension=8,
        layers=1,
        attention_heads=2,
        feed_forward_dimension=16,
        dropout=0.0,
        tie_embeddings=False,
    )


def _moe_configuration() -> MoeGptConfiguration:
    return MoeGptConfiguration(
        type=ModelType.MOE_GPT,
        dimension=8,
        layers=1,
        attention_heads=2,
        expert_feed_forward_dimension=16,
        expert_count=4,
        router_top_k=2,
        dropout=0.0,
        tie_embeddings=False,
    )


class _FixedOutputModel(nn.Module):
    def __init__(self, model_output: ModelOutput) -> None:
        super().__init__()
        self.model_output = model_output

    def forward(self, token_ids: torch.Tensor) -> ModelOutput:
        return self.model_output
