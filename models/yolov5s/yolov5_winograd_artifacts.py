"""Pinned v8 deployments: the same official weights/graphs with Winograd lowering.

Regenerate all six profiles through the compiler when changing the emitter.
"""

CANONICAL_WINOGRAD_ARTIFACTS = {
    "s": {
        "variant": "s",
        "format": "andromeda.yolov5s.single-bin",
        "graph_sha256": "0a3f6dbd9e3950fe5b13e829effeb5827c06321daee9d75f3fd690965939fc22",
        "weights_sha256": "273634fad085fa82ed76b5c5f03d80abebf3c8edcf4c8dbda95697c8c8b7d4d3",
        "model_sha256": "4b4a52f2f1d2b6507435215e7dae0f83946266da502df66c40380222d239ee75",
        "program_sha256": "43bbdfb4df11eae81641d96559fa2410f9c54227579a66b5a3b208c136a58e29",
        "manifest_sha256": "8d7d85409a397b96babecdecb6fdd00208948f82cf80a5a2191d86c807bf239b",
        "bundle_sha256": "0e722dc6ead45084866fc672863228233d8be5f2a87f83d2243d1f6f795fd079",
        "model_bytes": 10726208,
        "program_bytes": 783168
    },
    "n": {
        "variant": "n",
        "format": "andromeda.yolov5n.single-bin",
        "graph_sha256": "69dbe201b089756886973e98b9f24e8b58c27baf195f7b31c97f7d9c9c2e4272",
        "weights_sha256": "b8ca688e49a44c87b657de6b5c4c9d7b0193718c40ccc8f860733cd89865f666",
        "model_sha256": "ac9fb3e9605f3db6d9136a135178844e1f7142e06bbb7f90a610eeca24e38f81",
        "program_sha256": "ea04c37a9d4e44742fb9dd36610172fb1f266712d1978d2ec3e1d3bd6499c811",
        "manifest_sha256": "e09690e43c8762adfe86d8876550c26ce6f5a82e5c7fc820ef85f89488d06b2e",
        "bundle_sha256": "198b477a005646f74f1dc52b38f1a3af73a87f3a5fb568fa850dac7bf4726f58",
        "model_bytes": 1406208,
        "program_bytes": 99456
    }
}
