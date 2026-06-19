from pathlib import Path

from llm_lite.config.loading import load_experiment_configuration
from llm_lite.data.datasets import PackedSequence, PackedSequenceDataset
from llm_lite.model.gpt import DenseGpt
from llm_lite.tokenizer.character import train_character_tokenizer
from llm_lite.training.trainer import train_model


def test_training_checkpoint_resume(tmp_path: Path) -> None:
    experiment_configuration = load_experiment_configuration(
        configuration_path=Path("configs/verify_one_sentence.yaml"),
    )
    tokenizer = train_character_tokenizer(
        texts=["hello world\n"],
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )
    token_ids = tokenizer.encode(text="hello world\n", add_bos=True, add_eos=True)
    dataset = PackedSequenceDataset(sequences=[PackedSequence(token_ids=tuple(token_ids))])
    model = DenseGpt(
        model_configuration=experiment_configuration.model,
        vocabulary_size=tokenizer.vocabulary_size,
    )

    first_result = train_model(
        model=model,
        dataset=dataset,
        training_configuration=experiment_configuration.training,
        artifact_directory=tmp_path,
    )
    second_result = train_model(
        model=model,
        dataset=dataset,
        training_configuration=experiment_configuration.training,
        artifact_directory=tmp_path,
    )

    assert first_result.final_step == experiment_configuration.training.maximum_steps
    assert second_result.resumed_from_step == experiment_configuration.training.maximum_steps
