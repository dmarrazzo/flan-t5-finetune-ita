# Import objects from KubeFlow DSL Library
from kfp.dsl import (
    component,
    Output,
    Model,
)


@component(base_image="python:3.12",
           packages_to_install=["huggingface_hub"])
def fetch_model(
    model_name: str,
    model_version: str,
    allowed_patterns: str,
    original_model: Output[Model],
):
    """
    Downloads a model checkpoint from HuggingFace repository

    Args:
        model_name: name of the repository that contains the checpoint on HF
        model_version: checkpoint version
        original_model:  Artifact where the downloaded model checkpoint will be stored.
    """
    try:
        import os
        import zipfile
        from pathlib import Path
        import huggingface_hub as hf
    except Exception as e:
        raise e

    HF_TOKEN: str = os.getenv("HF_TOKEN")

    # Download model checkpoint from HuggingFace repositories
    local_path: str = "/".join(("/tmp/", model_name))
    os.makedirs(local_path, exist_ok=True)

    print(f"Downloading model checkpoint: {model_name}")
    model_path = hf.snapshot_download(repo_id=model_name,
                                    allow_patterns=allowed_patterns,
                                    revision=model_version,
                                    token=HF_TOKEN,
                                    local_dir=local_path)

    # save output dataset to S3
    original_model._set_path(original_model.path + ".zip")
    srcdir = Path(local_path)

    with zipfile.ZipFile(original_model.path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for entry in srcdir.rglob("*"):
            zip_file.write(entry, entry.relative_to(srcdir))