#!/usr/bin/env python

from kfp.v2.dsl import (
    component,
    Output,
    Dataset
)

@component(
    base_image="python:3.12",
    packages_to_install=["boto3"],          # install boto3 in the container
)
def download_tar_from_s3(
    dataset_name: str,
    dataset_version: str,
    file_name: str,
    output_tar: Output[Dataset],          # path where the tar will be written
):
    """
    Downloads a tar.gz file from an S3 bucket to the component output.

    Args:
        dataset_name: Name of the dataset (e.g. a folder under the dataset_bucket)
        dataset_version: Version of the dataset (also, a folder under the dataset_name path)
        file_name:   Full key/path of the tar.gz file inside the bucket.
        output_tar:  Artifact where the downloaded tar will be stored.
    """
    import boto3
    import os
    from botocore.exceptions import ClientError

    # get parameters from env
    s3_endpoint: str = os.getenv("AWS_S3_ENDPOINT")
    bucket_name: str = os.getenv("AWS_S3_BUCKET")

    # build object name to download
    object_name: str = f"{dataset_name}/{dataset_version}/{file_name}"

    # connect to S3 storage
    s3_client = boto3.client("s3", endpoint_url=s3_endpoint)

    try:
        s3_client.download_file(bucket_name, object_name, output_tar.path)
        print(f"Downloaded {object_name} from bucket {bucket_name} to {output_tar.path}")
    except ClientError as e:
        raise RuntimeError(
            f"Failed to download {object_name} from {bucket_name}: {e}"
        ) from e