# Import KubeFlow Pipelines library
import kfp

# Import objects from the DSL library
from kfp.dsl import pipeline
from kfp import kubernetes

# compile pipeline for debugging
from kfp import compiler

# Component imports
from download_from_s3 import download_tar_from_s3
from fetch_model import fetch_model
from persistent_data import unzip_data
from train_model import train_model
from convert_model import convert_model
from upload_model import push_to_model_registry

# Pipeline definition

# name of the data connection that points to the s3 model storage bucket
datasets_connection_secret_name = "s3-datasets"
data_connection_secret_name = "s3-models"
artifacts_connection_secret_name = "s3-artifacts"
huggingface_api_secret = "huggingface-secret"


# Create pipeline
@pipeline(
    name="flan-t5-anon-ita-finetune-pipeline",
    description="Finetune a FLAN-T5 Seq2Seq model to anonymize PII in italian language",
)
def training_pipeline(
    hyperparameters: dict,
    model_name: str,
    model_version: str,
    model_allowed_patterns: str,
    dataset_name: str,
    dataset_version: str,
    dataset_file_name: str,
    pipeline_version: str,
    author_name: str,
    registry: str,
    cluster_domain: str,
    dataset_path: str,
    model_path: str,
    finetuned_model_path: str
):

    dataset_fetch_task = download_tar_from_s3(
                            dataset_name=dataset_name,
                            dataset_version=dataset_version,
                            file_name=dataset_file_name
                        )
    kubernetes.use_secret_as_env(
        dataset_fetch_task,
        secret_name=datasets_connection_secret_name,
        secret_key_to_env={
            "AWS_S3_ENDPOINT": "AWS_S3_ENDPOINT",
            "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
            "AWS_S3_BUCKET": "AWS_S3_BUCKET",
            "AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION",
        },
    )

    # download base model from HF
    fetch_model_task = fetch_model(
        model_name=model_name, model_version=model_version,
        allowed_patterns=model_allowed_patterns
    )
    kubernetes.use_secret_as_env(
        fetch_model_task,
        secret_name=huggingface_api_secret,
        secret_key_to_env={
            "HF_TOKEN": "HF_TOKEN",
        },
    )

    # unzip model to persistent storage
    unzip_data_task = unzip_data(
        model_dir=model_path,
        dataset_dir=dataset_path,
        model=fetch_model_task.outputs["original_model"],
        dataset=dataset_fetch_task.outputs["output_tar"]
    )
    unzip_data_task.after(dataset_fetch_task)
    unzip_data_task.after(fetch_model_task)

    # mount persistent volume...
    kubernetes.mount_pvc(
        unzip_data_task,
        pvc_name="training",
        mount_path="/data",
    )

    # Train model
    train_model_task = train_model(dataset_dir=dataset_path,
                                   original_model_dir=model_path,
                                   finetuned_model_dir=finetuned_model_path,
                                   hyperparameters=hyperparameters)
    train_model_task.after(unzip_data_task)
    # train_model_task.set_cpu_limit("1")
    # train_model_task.set_memory_limit("4G")

    # mount persistent volume...
    kubernetes.mount_pvc(
        train_model_task,
        pvc_name="training",
        mount_path="/data",
    )

    # convert to onnx
    convert_task = convert_model(
        checkpoint_dir=finetuned_model_path,
        finetuned_model=train_model_task.outputs["finetuned_model"]
    )
    convert_task.after(train_model_task)
    # convert_task.set_cpu_limit("1")
    # convert_task.set_memory_limit("4G")

    # mount persistent volume...
    kubernetes.mount_pvc(
        convert_task,
        pvc_name="training",
        mount_path="/data",
    )

    # push to model registry
    push_task = push_to_model_registry(model_name=model_name,
                                       finetuned_model=train_model_task.outputs["finetuned_model"],
                                       version=dataset_version,
                                       registry=registry,
                                       cluster_domain=cluster_domain,
                                       author_name=author_name,
                                       data_path=finetuned_model_path)
    push_task.after(train_model_task)
    push_task.after(convert_task)
    kubernetes.use_secret_as_env(
        push_task,
        secret_name=data_connection_secret_name,
        secret_key_to_env={
            "AWS_S3_ENDPOINT": "AWS_S3_ENDPOINT",
            "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
            "AWS_S3_BUCKET": "AWS_S3_BUCKET",
            "AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION",
        },
    )
    # mount persistent volume...
    kubernetes.mount_pvc(
        push_task,
        pvc_name="training",
        mount_path="/data",
    )


# start pipeline
if __name__ == "__main__":
    metadata = {
        "hyperparameters": {
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "max_length": 256,
            "optimizer": "AdamW",
            "train_val_split": 0.8
        },
        "model_name": "google/flan-t5-small",
        "model_version": "main",
        "model_allowed_patterns": "*",
        "dataset_name": "ita_dataset",
        "dataset_version": "1",
        "dataset_file_name": "ita_dataset.tar.gz",
        "pipeline_version": "1",
        "author_name": "DevOps Team",
        "registry": "model-registry-rest",
        "cluster_domain": "apps.ocp.lab",
        "model_path": "/data/model",
        "dataset_path": "/data/dataset",
        "finetuned_model_path": "/data/finetuned"
    }

    namespace_file_path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
    with open(namespace_file_path, "r") as namespace_file:
        namespace = namespace_file.read()

    kubeflow_endpoint = f"https://ds-pipeline-dspa.{namespace}.svc:8443"

    sa_token_file_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    with open(sa_token_file_path, "r") as token_file:
        bearer_token = token_file.read()

    ssl_ca_cert = "/var/run/secrets/kubernetes.io/serviceaccount/service-ca.crt"

    # compile pipeline to kubernetes yaml
    compiler.Compiler().compile(training_pipeline, "finetune-pipeline.yaml")

    # Run pipeline on cluster
    print(f"Connecting to Data Science Pipelines: {kubeflow_endpoint}")
    client = kfp.Client(
        host=kubeflow_endpoint, existing_token=bearer_token, ssl_ca_cert=ssl_ca_cert
    )

    # run ad-hoc pipeline
    client.create_run_from_pipeline_func(
        training_pipeline,
        arguments=metadata,
        experiment_name="flan-t5-anon-ita-finetune-pipeline",
        enable_caching=True,
    )

