import kfp

def cascade_delete_pipeline(client, pipeline_name):
    try:
        # 1. Get the Pipeline ID
        pipeline_id = client.get_pipeline_id(pipeline_name)
        if not pipeline_id:
            print(f"Pipeline '{pipeline_name}' not found.")
            return

        # 2. List and Delete all Versions
        # In V2, the attribute is .pipeline_versions
        versions_response = client.list_pipeline_versions(pipeline_id=pipeline_id)
        
        # Check if the list exists and is not empty
        if hasattr(versions_response, 'pipeline_versions') and versions_response.pipeline_versions:
            for version in versions_response.pipeline_versions:
                print(f"Deleting version: {version.display_name} ({version.pipeline_version_id})")
                client.delete_pipeline_version(pipeline_id, version.pipeline_version_id)
                
        else:
            print("No versions found to delete.")

        # 3. Delete the Parent Pipeline
        print(f"Deleting pipeline container: {pipeline_name} ({pipeline_id})")
        client.delete_pipeline(pipeline_id)

        # 4. Cleanup Related Experiments
        exp_response = client.list_experiments(page_size=100)
        
        # In V2, the attribute is .experiments
        if hasattr(exp_response, 'experiments') and exp_response.experiments:
            for exp in exp_response.experiments:
                # Be careful with the match logic
                if pipeline_name in exp.display_name:
                    print(f"Deleting matching experiment: {exp.display_name} ({exp.experiment_id})")
                    client.delete_experiment(exp.experiment_id)

        print("--- Cascade deletion complete ---")

    except Exception as e:
        print(f"Error during deletion: {e}")

def delete_runs_by_name(client, run_name_to_delete):
    """
    Finds and deletes all runs that match a specific display name.
    """
    try:
        print(f"Searching for runs named: '{run_name_to_delete}'...")
        
        # 1. Fetch runs (using page_size to ensure we see recent ones)
        # In KFP v2, the response attribute is .runs
        runs_response = client.list_runs(page_size=100)
        
        if not hasattr(runs_response, 'runs') or not runs_response.runs:
            print("No runs found in the cluster.")
            return

        deleted_count = 0
        for run in runs_response.runs:
            # 2. Check for an exact match or partial match
            if run.display_name == run_name_to_delete:
                print(f"Deleting run: {run.display_name} (ID: {run.run_id})")
                
                # 3. Perform the deletion
                client.delete_run(run.run_id)
                deleted_count += 1
        
        if deleted_count == 0:
            print(f"No runs found matching the name '{run_name_to_delete}'.")
        else:
            print(f"Successfully deleted {deleted_count} run(s).")

    except Exception as e:
        print(f"Error during run deletion: {e}")


if __name__ == "__main__":

    PIPELINE_NAME = "flan-t5-finetune"
    namespace_file_path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
    with open(namespace_file_path, "r") as namespace_file:
        namespace = namespace_file.read()

    kubeflow_endpoint = f"https://ds-pipeline-dspa.{namespace}.svc:8443"

    sa_token_file_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    with open(sa_token_file_path, "r") as token_file:
        bearer_token = token_file.read()

    ssl_ca_cert = "/var/run/secrets/kubernetes.io/serviceaccount/service-ca.crt"

    print(f"Connecting to Data Science Pipelines: {kubeflow_endpoint}")

    client = kfp.Client(
         host=kubeflow_endpoint, existing_token=bearer_token, ssl_ca_cert=ssl_ca_cert
    )

    # Use the name of the pipeline you want to wipe out
    cascade_delete_pipeline(client, PIPELINE_NAME)
    
    #delete_runs_by_name(client, "flan-t5-finetune-run")