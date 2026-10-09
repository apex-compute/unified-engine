# VeraPulse controller-private vision weights

The existing multi-engine vision stage now uses private copies of its six bf16
projections per layer: Q, K, V, O, FC1 and FC2. Norms, biases, activations, prefix
weights and action-expert weights retain their existing placement. Each vision
engine emits its own projection addresses, preserving the existing row split,
camera batching, arithmetic and worker-program allocation.

Copies occupy free Alveo controller windows above the fixed low-4-GiB model.
U50 supports eight 512 MiB windows; 16 GiB U55C supports twelve separate 1 GiB
regions. The 8 GiB U55C image has four free controllers, shared cyclically by
additional engines. Kintex keeps shared weights because this fixed layout has
no free external reserve.

Program manifests save the actual replica windows and source/destination byte
ranges. Bin loading restores replicas and rejects missing or changed layouts
before loading programs; regenerate existing Alveo multi-engine bins. The
existing `--engines N` / `--vis_8` options select the vision path. Offline tests
cover copy fidelity, projection sizes, distinct addresses, capacity and stale
cache rejection. Hardware action fidelity and throughput remain unmeasured for
this change.
