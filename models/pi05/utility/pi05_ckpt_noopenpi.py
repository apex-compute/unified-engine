"""openpi-free provider for the two symbols the pi05_libero weight export needs.

openpi is only ever touched by the ONE-TIME weights_export/ materialization --
never by inference, which runs from params.bin on the FPGA. And neither symbol
the exporter uses actually needs openpi's model code:

  openpi.shared.download.maybe_download(url)
      -> `gsutil -m cp -r` of a PUBLIC GCS bucket into <cache root>/<host>/<path>.
         Replaced here with anonymous HTTPS GETs against storage.googleapis.com,
         writing to the exact same cache path -- so if the real openpi is ever
         installed it finds this cache and skips the download (and vice versa).

         The cache root is OPENPI_DATA_HOME (openpi's own env override), which the
         callers set to the model's <model>_bin/ dir before calling in -- so the
         checkpoint lands beside the model like every other model's weights, not in
         a machine-global ~/.cache. The ~/.cache/openpi default below is only the
         fallback for a direct call that never set the variable.

  openpi.models.model.restore_params(path, restore_type=np.ndarray)
      -> a plain orbax PyTreeCheckpointer restore plus a strip of the nnx "value"
         leaf suffix. Replaced here with the same orbax calls, verbatim.

Everything downstream -- _flatten_params, _canonical_name, the .npy writes,
manifest.json, the completeness check -- stays the original code in
pi05_weight_export.py / pi05_test.py, so the export produced through this
path is byte-identical to the one openpi would have produced.

Both exporters call _import_openpi(), which tries the real openpi first and falls
back to this module, so the auto-download-on-first-run flow is unchanged: just run
`python pi05_test.py`.

Requires: numpy, jax, orbax-checkpoint, flax (already in the engine env).
Does NOT require: openpi, lerobot, jax[cuda12], wandb, uv.

Can also be run directly to do only the export step:
    python pi05_ckpt_noopenpi.py
"""

import concurrent.futures
import json
import os
import pathlib
import shutil
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

_GCS_JSON_API = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"
_GCS_MEDIA = "https://storage.googleapis.com/{bucket}/{obj}"
DEFAULT_CACHE_DIR = "~/.cache/openpi"          # same default as openpi
_OPENPI_DATA_HOME = "OPENPI_DATA_HOME"         # same env override as openpi


def _cache_dir():
    d = pathlib.Path(os.getenv(_OPENPI_DATA_HOME, DEFAULT_CACHE_DIR)).expanduser().resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _list_bucket(bucket, prefix):
    """List every object under gs://bucket/prefix via the anonymous JSON API."""
    items, token = [], None
    while True:
        query = {"prefix": prefix, "maxResults": "1000"}
        if token:
            query["pageToken"] = token
        url = _GCS_JSON_API.format(bucket=bucket) + "?" + urllib.parse.urlencode(query)
        with urllib.request.urlopen(url, timeout=60) as resp:
            page = json.load(resp)
        for item in page.get("items", []):
            if not item["name"].endswith("/"):     # skip directory placeholders
                items.append((item["name"], int(item["size"])))
        token = page.get("nextPageToken")
        if not token:
            return items


def _stat_object(bucket, obj):
    """Metadata for a single object, or None if it is not one (i.e. a dir prefix)."""
    url = (_GCS_JSON_API.format(bucket=bucket) + "/"
           + urllib.parse.quote(obj, safe=""))
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return None
        raise


def _fetch_object(bucket, obj, dest, size):
    """Download one object to dest. No-op if dest already has the exact size."""
    if dest.exists() and dest.stat().st_size == size:
        print(f"[dl]   cached  {obj}  ({size / 1e6:.1f} MB)", flush=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".partial")
    url = _GCS_MEDIA.format(bucket=bucket, obj=urllib.parse.quote(obj))
    with urllib.request.urlopen(url, timeout=600) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out, length=8 << 20)
    got = tmp.stat().st_size
    if got != size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"[dl] size mismatch for {obj}: got {got}, expected {size}")
    tmp.replace(dest)
    print(f"[dl]   done    {obj}  ({size / 1e6:.1f} MB)", flush=True)


def maybe_download(url, *, force_download=False, **kwargs):
    """Anonymous-HTTPS stand-in for openpi.shared.download.maybe_download.

    Handles local paths and gs:// URLs on public buckets -- which is all the
    pi05_libero checkpoint needs. Caches to the SAME path openpi would use:
    $OPENPI_DATA_HOME/<netloc>/<path>, where callers point OPENPI_DATA_HOME at the
    model's <model>_bin/ dir. Re-running is idempotent: objects already present at
    the right size are skipped, so a partial download resumes.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "":
        path = pathlib.Path(url)
        if not path.exists():
            raise FileNotFoundError(f"File not found at {url}")
        return path.resolve()
    if parsed.scheme != "gs":
        raise ValueError(f"[dl] only gs:// URLs are supported by this stand-in, got: {url}")

    bucket, prefix = parsed.netloc, parsed.path.strip("/")
    local_path = (_cache_dir() / bucket / prefix).resolve()

    # A single object (e.g. gs://big_vision/paligemma_tokenizer.model) rather than a
    # directory prefix -- openpi's maybe_download handles both, so this must too.
    meta = _stat_object(bucket, prefix)
    if meta is not None:
        size = int(meta["size"])
        if local_path.is_file() and local_path.stat().st_size == size and not force_download:
            return local_path
        print(f"[dl] {prefix}  {size / 1e6:.1f} MB  ->  {local_path}", flush=True)
        _fetch_object(bucket, prefix, local_path, size)
        return local_path

    try:
        objects = _list_bucket(bucket, prefix + "/")
    except OSError as err:
        # Offline but already downloaded: use what we have rather than hard-failing.
        if local_path.exists():
            print(f"[dl] listing failed ({err}); using existing {local_path}", flush=True)
            return local_path
        raise
    if not objects:
        raise RuntimeError(f"[dl] no objects found under {url}")
    total = sum(size for _, size in objects)
    print(f"[dl] {len(objects)} objects, {total / 1e9:.2f} GB  ->  {local_path}", flush=True)

    # Size the guard by what is still MISSING, so a fully-cached checkpoint never
    # trips it on a nearly-full disk.
    def _present(name, size):
        f = local_path / name[len(prefix) + 1:]
        return f.exists() and f.stat().st_size == size

    need = sum(size for name, size in objects if not _present(name, size))
    free = shutil.disk_usage(_cache_dir()).free
    if need and free < need * 1.05:
        raise RuntimeError(
            f"[dl] not enough free disk: need ~{need / 1e9:.1f} GB, "
            f"have {free / 1e9:.1f} GB free. (The export then needs ~13 GB more.)")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(_fetch_object, bucket, name, local_path / name[len(prefix) + 1:], size)
            for name, size in objects
        ]
        for fut in concurrent.futures.as_completed(futures):
            fut.result()        # re-raise the first failure

    print(f"[dl] checkpoint ready: {local_path}", flush=True)
    return local_path


def restore_params(params_path, *, restore_type=np.ndarray, dtype=None, sharding=None):
    """Orbax-only stand-in for openpi.models.model.restore_params.

    Same orbax calls as upstream; upstream's jax.Array/sharding branch is dropped
    because the exporter always asks for restore_type=np.ndarray.
    """
    import jax
    import orbax.checkpoint as ocp
    from flax import traverse_util

    if restore_type is not np.ndarray:
        raise ValueError(
            f"[export] this stand-in only restores as np.ndarray, got {restore_type}")

    params_path = pathlib.Path(params_path).resolve()

    with ocp.PyTreeCheckpointer() as ckptr:
        # orbax <=0.11 (what openpi pins) returns the tree metadata directly;
        # orbax >=0.12 wraps it in a StepMetadata whose .item_metadata holds it.
        metadata = ckptr.metadata(params_path)
        tree_metadata = getattr(metadata, "item_metadata", None)
        if tree_metadata is None:
            tree_metadata = metadata
        item = {"params": tree_metadata["params"]}
        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(
                        sharding=sharding, restore_type=restore_type, dtype=dtype),
                    item,
                ),
            ),
        )["params"]

    # Checkpoints saved via nnx.State end every key path with "value"; drop it.
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)


def import_openpi_fallback():
    """(maybe_download, restore_params) in the shape _import_openpi() returns."""
    return maybe_download, restore_params


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pi05_weight_export as wexp

    print(wexp.ensure_weights_export())
