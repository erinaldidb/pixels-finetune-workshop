#!/usr/bin/env python3
"""Register a fine-tuned NV-Segment model in Unity Catalog under the practitioner's schema.

Creates <catalog>.ws_<user> if needed, registers the run's MLflow model as
<catalog>.ws_<user>.<model_name>, sets an alias and tags. Progress goes to
stderr; a JSON report goes to stdout. Run from a Databricks notebook (CPU).

    python scripts/register_model.py --run-id RUN_ID --catalog CATALOG \
        [--schema ws_<user>] [--model-name nv_segment_ctmr_imagechd] [--alias workshop] [--dry-run]

From a notebook, call it in-process so it uses the notebook's credentials:
    import sys; sys.path.insert(0, f"{SKILL_DIR}/scripts")
    from register_model import register_run
    register_run(RUN_ID, CATALOG)            # schema defaults to ws_<user>

Exit code 0 = registered (or dry run ok), 1 = failed, 2 = bad arguments.
"""
import argparse
import json
import re
import sys

IDENT_RE = re.compile(r"^[a-z0-9_]{1,255}$")  # UC names used here: lowercase letters, digits, underscore


def sanitize(name: str) -> str:
    """'First.Last@company.com' -> 'first_last' (UC-safe identifier)."""
    local = name.split("@", 1)[0].lower()
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]", "_", local)).strip("_")


def default_schema(user_name: str) -> str:
    return f"ws_{sanitize(user_name)}"


def default_model_name(tags: dict) -> str:
    model_id = tags.get("segment.model_id", "nv_segment")
    return sanitize(model_id.replace("-", "_")) + "_finetuned"


def log(msg: str):
    print(msg, file=sys.stderr)


def ensure_schema(catalog: str, schema: str) -> str:
    """Create the schema if missing; returns 'created' or 'exists'."""
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.errors import NotFound, PermissionDenied, ResourceAlreadyExists

    w = WorkspaceClient()
    try:
        w.schemas.get(f"{catalog}.{schema}")
        return "exists"
    except NotFound:
        pass
    try:
        w.schemas.create(name=schema, catalog_name=catalog,
                         comment="NVIDIA MedTech fine-tuning workshop schema")
        return "created"
    except ResourceAlreadyExists:
        return "exists"
    except PermissionDenied as exc:
        raise SystemExit(f"Cannot create {catalog}.{schema}: {exc}. Ask the instructor for "
                         f"USE CATALOG and CREATE SCHEMA on {catalog}.")


def resolve_model_uri(client, run_id: str) -> tuple:
    run = client.get_run(run_id)
    tags = run.data.tags
    uri = tags.get("segment.model_uri") or f"runs:/{run_id}/model"
    if tags.get("status") != "SUCCESS":
        raise SystemExit(f"run {run_id} has status {tags.get('status')!r}; register a successful run")
    return uri, tags, run.data.params, run.data.metrics


def register(args) -> dict:
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri("databricks")
    tracking = MlflowClient()
    uri, tags, params, metrics = resolve_model_uri(tracking, args.run_id)
    if args.schema is None:
        from databricks.sdk import WorkspaceClient
        args.schema = default_schema(WorkspaceClient().current_user.me().user_name)
    model_name = args.model_name or default_model_name(tags)
    for kind, value in (("catalog", args.catalog), ("schema", args.schema), ("model name", model_name)):
        if not IDENT_RE.match(value):
            raise SystemExit(f"invalid {kind} {value!r}: use lowercase letters, digits and underscores")
    full_name = f"{args.catalog}.{args.schema}.{model_name}"
    result = {"run_id": args.run_id, "model_uri": uri, "registered_model": full_name, "alias": args.alias}
    if args.dry_run:
        return {**result, "dry_run": True}

    schema_status = ensure_schema(args.catalog, args.schema)
    log(f"Schema {args.catalog}.{args.schema}: {schema_status}")
    mlflow.set_registry_uri("databricks-uc")
    # launcher params: segment.model_id / workflow / repo_commit / hf_revision / num_gpus
    version_tags = {k.split(".", 1)[1]: v for k, v in params.items() if k.startswith("segment.")}
    version_tags["source_run_id"] = args.run_id
    for key in ("eval_before_mean_dice", "eval_after_mean_dice", "best_val_mean_dice"):
        if key in metrics:
            version_tags[key] = f"{metrics[key]:.4f}"
    log(f"Registering {uri} as {full_name} ...")
    mv = mlflow.register_model(uri, full_name, tags=version_tags)
    uc = MlflowClient(registry_uri="databricks-uc")
    uc.set_registered_model_alias(full_name, args.alias, mv.version)
    uc.update_registered_model(full_name, description=(
        f"NVIDIA {version_tags.get('model_id', 'NV-Segment')} fine-tuned with the "
        f"{version_tags.get('workflow', '')} workflow on AI Runtime (run {args.run_id}). "
        "Engineering/research use only; not for clinical use."))
    return {**result, "version": mv.version, "schema_status": schema_status}


def register_run(run_id: str, catalog: str, schema: str | None = None, model_name: str | None = None,
                 alias: str = "workshop", dry_run: bool = False) -> dict:
    """In-process entry point for notebooks (inherits notebook auth, unlike a subprocess)."""
    args = argparse.Namespace(run_id=run_id, catalog=catalog, schema=schema, model_name=model_name,
                              alias=alias, dry_run=dry_run)
    return register(args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True, help="MLflow run ID of a successful fine-tuning job")
    ap.add_argument("--catalog", required=True, help="Workshop catalog, e.g. pixels_workshop_catalog")
    ap.add_argument("--schema", default=None, help="Default: ws_<your user name>")
    ap.add_argument("--model-name", default=None, help="Default: <model_id>_finetuned")
    ap.add_argument("--alias", default="workshop", help="Alias pointing at the new version")
    ap.add_argument("--dry-run", action="store_true", help="Resolve names and the model URI only")
    args = ap.parse_args()
    result = register(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
