# Import objects from KubeFlow DSL Library
from kfp.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Artifact,
    Model
)


@component(base_image="python:3.12")
def unzip_data(
    model_dir: str,
    dataset_dir: str,
    model: Input[Model],
    model_properties: Output[Artifact],
    dataset: Input[Dataset],
    dataset_properties: Output[Artifact]
):
    # import zipfile lib
    import json
    import os
    import tarfile
    from pathlib import PosixPath
    from zipfile import ZipFile

    # make sure directory exists
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    # 1. decompress model to disk
    with ZipFile(model.path, 'r') as compressed_model:
        compressed_model.extractall(model_dir)

    # save model properties
    m_props = {
        "model_filename": model.path,
        "model_architecture": "Seq2Seq",
        "model_type": "T5",
        "framework": "torch",
        "flavor": "small",
    }

    model_properties.path += ".json"
    with open(model_properties.path, "w") as artifact_dump:
        json.dump(m_props, artifact_dump)

    # 2. decompress dataset to disk
    with tarfile.open(dataset.path) as compressed_dataset:
        compressed_dataset.extractall(dataset_dir)

    # save dataset properties
    d_props = {
        "dataset_filename": model.path,
        "dataset_format": "json",
        "framework": "torch",
    }

    dataset_properties.path += ".json"
    with open(dataset_properties.path, "w") as artifact_dump:
        json.dump(d_props, artifact_dump)
