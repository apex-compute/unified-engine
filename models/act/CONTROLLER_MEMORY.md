# ACT controller-private weights

The existing multi-engine path now reads immutable bf16 weights from Alveo
controller replicas in free memory above 4 GiB. This covers both the central
matrix-multiply wrapper and the SRAM convolution kernel's weight-strip loads.
Activation/attention operands remain at their shared addresses, and the existing
row/column splits and four-phase synchronization stay in place.

U50 offers eight 512 MiB private windows in the upper stack. A 16 GiB U55C can
assign one free 1 GiB region per engine. An 8 GiB U55C has four free controller
regions after reserving the original low-4-GiB model; more engines share those
four copies. Kintex retains shared source addresses because its fixed layout
leaves no external reserve.

The existing `--engines N` option selects this behavior. Cached program manifests
include the replica geometry and byte-copy records; loading restores the actual
weights before reusing instructions. Stale maps trigger a rebuild. Offline tests
cover exact bytes, guards, cache replay, matrix operands, and convolution strip
addresses. Hardware model throughput and golden-output comparisons remain pending.
