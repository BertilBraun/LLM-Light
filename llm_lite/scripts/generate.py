import argparse
from pathlib import Path

from llm_lite.config.loading import load_experiment_configuration
from llm_lite.config.models import InferenceConfiguration
from llm_lite.inference.engine import generate_text
from llm_lite.model.gpt import DenseGpt
from llm_lite.pipeline.registry import ArtifactRegistry
from llm_lite.pipeline.stage import StageName
from llm_lite.tokenizer.loading import load_tokenizer
from llm_lite.training.checkpoint import load_latest_checkpoint


def build_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--config", required=True, type=Path)
    argument_parser.add_argument("--prompt", required=True)
    argument_parser.add_argument("--maximum-new-tokens", type=int)
    return argument_parser


def main() -> int:
    arguments = build_argument_parser().parse_args()
    experiment_configuration = load_experiment_configuration(
        configuration_path=arguments.config,
    )
    maximum_new_tokens = (
        experiment_configuration.inference.maximum_new_tokens
        if arguments.maximum_new_tokens is None
        else arguments.maximum_new_tokens
    )
    registry = ArtifactRegistry(run_directory=experiment_configuration.experiment.output_dir)
    tokenizer = load_tokenizer(
        directory=registry.artifact_directory(StageName.TOKENIZER.value),
        tokenizer_configuration=experiment_configuration.tokenizer,
    )
    model = DenseGpt(
        model_configuration=experiment_configuration.model,
        vocabulary_size=tokenizer.vocabulary_size,
    )
    checkpoint_step = load_latest_checkpoint(
        checkpoint_directory=registry.artifact_directory(StageName.PRETRAINING.value)
        / "checkpoints",
        model=model,
        optimizer=None,
    )
    if checkpoint_step is None:
        raise ValueError("Generation requires a completed pretraining checkpoint.")
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=arguments.prompt,
        inference_configuration=InferenceConfiguration(
            engine=experiment_configuration.inference.engine,
            precision=experiment_configuration.inference.precision,
            quantization=experiment_configuration.inference.quantization,
            decoding=experiment_configuration.inference.decoding,
            maximum_new_tokens=maximum_new_tokens,
        ),
    )
    print(generated_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
