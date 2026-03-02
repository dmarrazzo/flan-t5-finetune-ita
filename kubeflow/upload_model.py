# Import KubeFlow Pipelines library
from kfp.dsl import (
    component,
    Input,
    Model
)


@component(base_image="python:3.12",
           packages_to_install=['pip==24.2',
                                'setuptools==74.1.3',
                                'boto3==1.36.12',
                                'model-registry',
                                'debugpy'])
def push_to_model_registry(
    model_name: str,
    finetuned_model: Input[Model],
    version: str,
    registry: str,
    cluster_domain: str,
    author_name: str,
    data_path: str
):
    from model_registry import ModelRegistry
    from model_registry import utils
    import boto3
    import sys
    import threading
    from botocore.exceptions import ClientError
    from pathlib import Path
    from zipfile import ZipFile
    import os

    import debugpy
    # Allow the debugger to attach on port 5678
    debugpy.listen(("0.0.0.0", 5678))
    print("Waiting for debugger attach...")
    debugpy.wait_for_client() # The code pauses here until you attach
    breakpoint() # Optional: Forces the debugger to stop here

    # prepare workdir
    WORKDIR: str = f"{data_path}/scratch"
    os.makedirs(WORKDIR, exist_ok=True)

    # decompress finetuned model
    with ZipFile(finetuned_model.path, 'r') as ftuned:
        ftuned.extractall(WORKDIR)

    # environment setup
    from os import environ
    environ["KF_PIPELINES_SA_TOKEN_PATH"] = "/var/run/secrets/kubernetes.io/serviceaccount/token"

    # Set up the model registry connection
    if not registry:
        registry = os.environ.get("REGISTRY_NAME")
    if not cluster_domain:
        cluster_domain = os.environ.get("REGISTRY_CLUSTER_DOMAIN")

    model_registry_url = f"https://{registry}.{cluster_domain}"

    # S3 parameters
    minio_endpoint = os.environ.get("AWS_S3_ENDPOINT")
    minio_accesskey = os.environ.get("AWS_ACCESS_KEY_ID")
    minio_secretkey = os.environ.get("AWS_SECRET_ACCESS_KEY")
    s3_model_bucket = os.environ.get('AWS_S3_BUCKET')
    s3_region = os.environ.get("AWS_DEFAULT_REGION")

    # registry connection object
    registry = ModelRegistry(server_address=model_registry_url,
                             port=443, author=author_name, is_secure=False)

    # Model details we want to register
    registered_model_name = model_name
    s3_model_prefix = f"{registered_model_name}"
    version = version

    # remote S3 paths
    s3_region = s3_region,
    s3_prefix = f"{s3_model_prefix}/torch/{version}"

    # connect to S3
    try:
        s3_client = boto3.client("s3",
                                 endpoint_url=minio_endpoint,
                                 aws_access_key_id=minio_accesskey,
                                 aws_secret_access_key=minio_secretkey)
    except Exception as e:
        raise Exception(f"Cannot instantiate S3 Client: {e}")

    # checks whether a file exists in a remote bucket
    def check_exists(s3api, bucket, filename):
        rsp = s3api.list_objects_v2(Bucket=bucket, Prefix=filename)
        try:
            contents = rsp.get("Contents")
            files = [obj.get("Key") for obj in contents]
            if filename in files:
                return True
            else:
                return False
        except Exception:
            return False

    # shamelessly stolen from aws docs :D
    class ProgressPercentage(object):
        def __init__(self, filename):
            self._filename = filename
            self._size = float(os.path.getsize(filename))
            self._seen_so_far = 0
            self._lock = threading.Lock()

        def __call__(self, bytes_amount):
            # To simplify, assume this is hooked up to a single filename
            with self._lock:
                self._seen_so_far += bytes_amount
                percentage = (self._seen_so_far / self._size) * 100
                sys.stdout.write(
                    "\r%s  %s / %s  (%.2f%%)" % (
                        self._filename, self._seen_so_far, self._size,
                        percentage))
                sys.stdout.flush()

    # resolve model data path for recursive upload
    model_data_path = Path(WORKDIR).resolve()

    # upload recursively
    for root, _, files in os.walk(model_data_path):
        for filename in files:
            local_path = Path(root) / filename
            # Compute the path relative to the base directory
            relative_path = local_path.relative_to(model_data_path)
            # Build the S3 key
            s3_key = f"{s3_prefix}/{relative_path.as_posix()}"

            # upload
            try:
                # Set the desired multipart threshold value (5GB)
                GB = 1024 ** 3
                transfer_config = boto3.s3.transfer.TransferConfig(multipart_threshold = 5*GB,
                                                                   use_threads=False)

                if not check_exists(s3_client,
                                    s3_model_bucket,
                                    str(local_path)):
                    s3_client.upload_file(
                        Filename=str(local_path),
                        Bucket=s3_model_bucket,
                        Key=s3_key,
                        Callback=ProgressPercentage(str(local_path)),
                        Config=transfer_config
                    )
                else:
                    print(f"File {str(local_path)} already exists in {s3_model_bucket}")

                print(f"Uploaded: {local_path} to s3://{s3_model_bucket}/{s3_key}")
            except ClientError as e:
                print(f"Failed to upload {local_path}: {e}")
            except Exception as e:
                print(f"Caught exception: {e}")

    # upload function
    def register(model_name, uri,
                 model_format_name, author, model_format_version,
                 model_version, storage_key, storage_path, description,
                 metadata):
        try:
            # register onnx model
            registered_model = registry.register_model(
                uri=uri,
                name=model_name,
                model_format_name=model_format_name,
                author=author,
                model_format_version=model_format_version,
                version=model_version,
                storage_key=storage_key,
                storage_path=storage_path,
                description=description,
                metadata=metadata
            )
            print(f"'{registered_model}' - '{model_name}' version '{model_version}'\n URL: https://rhods-dashboard-redhat-ods-applications.{cluster_domain}/modelRegistry/registry/registeredModels/1/versions/{registry.get_model_version(model_name, model_version).id}/details")
        except Exception as e:
            raise Exception(f"Exception during model registration: {e}")

    # now register checkpoint to model registry
    model: dict = {
            "model_name": registered_model_name,
            "uri": utils.s3_uri_from(f"{s3_prefix}",
                                     f"{s3_model_bucket}",
                                     endpoint=minio_endpoint),
            "author": author_name,
            "model_format_name": "torch",
            "model_format_version": "1",
            "model_version": f"{version}",
            "storage_key": "s3-models",
            "storage_path": f"{s3_prefix}",
            "description": f"TORCH Model Version {version}",
            "metadata": {
                        "format": "torch",
                        "license": "apache-2.0"
                    }
        }

    print(f"Registering: {model.get('model_version')}...")
    register(model_name=model.get('model_name'),
             uri=model.get("uri"),
             model_format_name=model.get('model_format_name'),
             author=model.get('author'),
             model_format_version=model.get('model_format_version'),
             model_version=model.get('model_version'),
             storage_key=model.get("storage_key"),
             storage_path=model.get('storage_path'),
             description=model.get('description'),
             metadata=model.get('metadata'))

    print("Model registered successfully")