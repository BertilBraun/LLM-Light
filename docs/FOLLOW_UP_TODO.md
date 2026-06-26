# Orchestration Follow-Up TODO

These tasks are related to the orchestration rewrite, but they can land
separately because they do not block the first local executor.

## Max Checkpoints

- Add `training.max_checkpoints` to the experiment config.
- Keep the latest `max_checkpoints` interval checkpoints plus the final
  checkpoint.
- Apply retention to both single-file checkpoints and sharded checkpoint
  directories.
- Delete only complete older checkpoints after a newer checkpoint has been
  written successfully.
- Update training checkpoint tests for full and sharded checkpoint retention.

## Export Stage

- Add `export` as a first-class stage name.
- Add export config fields for bundle location and optional TensorBoard
  inclusion.
- Make export read `runs/<experiment>/run_manifest.json`.
- Copy referenced artifact-store entries into the export bundle instead of
  assuming large artifacts live under `runs/<experiment>/artifacts`.
- Write a bundle manifest listing included artifacts, fingerprints, files, and
  run metadata.
- Update the existing export script/tests to use the run manifest and artifact
  store layout.
