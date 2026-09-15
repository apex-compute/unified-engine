# BigCodec background-noise check

**Background noise remaining is consistent with BigCodec's intended function.** [BigCodec](https://github.com/Aria-K-Alethia/BigCodec) is a speech codec trained to compress and reconstruct audio. Its [paper](https://arxiv.org/abs/2409.05377) describes reconstruction, adversarial and quantization objectives, with no dedicated noisy-to-clean enhancement objective. Use [DPDFNet](../../../../dpdfnet/validation/20260914_noisy20s/README.md) for noise suppression.

## Implementation check

The port includes the official encoder, vector quantizer and decoder. No noise-suppression module was omitted. The fixed low-pass filters around Snake activations control resampling/activation aliasing; the FPGA includes both filters. Importing DPDFNet's scalar DMA engine in the BigCodec runner reuses board I/O, not the DPDFNet neural model.

On 2026-09-15, a fresh CPU run used the literal pinned upstream `inference.py`, changing only its three CUDA placements to CPU. The full bus input contains 378,916 samples at 16 kHz (23.68225 s), padded to 379,000 samples. Source and checkpoint hashes were checked against the pinned manifest.

- Upstream and wrapper encoder features, direct quantizer tensors, and all **1,895 token IDs match exactly**.
- Re-embedding those tokens differs slightly from upstream's direct quantizer tensor because its straight-through expression introduces FP32 rounding. The resulting waveform difference is **0.00006988% relative L2**, with maximum absolute sample difference **7.90 × 10⁻⁷**.
- The fresh wrapper waveform samples and token IDs match the frozen CPU reference exactly.
- Upstream retains 84 padding samples and writes PCM-16. The wrapper restores the original length and writes float audio. The neural comparison uses floating-point samples before file encoding.

[CPU parity measurements, source/checkpoint hashes and output hashes](cpu_parity.json) preserve the full check. The [CPU reproducer](reproduce_cpu_parity.py) defaults to a preview; `--execute` runs the same neural comparison and saves fresh artifacts under the ignored `bigcodec_bin/` directory. It needs the pinned checkpoint, CPU core 7 and the BigCodec dependencies; it verifies cached upstream sources or fetches their pinned versions. The original measured generator remains identified by hash in the JSON.

These checks validate the CPU reference and intended graph. FPGA precision remains a separate limitation: the selected BF16 bus output differs from CPU by **20.55% relative L2**, and the eight-file pooled error is **23.94%**. Neither percentage measures noise removal.

## Same recording against clean speech

Scores below were recomputed against the paired clean bus reference using the existing audio evaluator. All samples are included without fitted delay or gain. DPDFNet's runner already compensates its fixed model delay. Higher scores are better, but the metrics combine noise and speech distortion; SI-SDR is particularly sensitive to codec waveform changes.

| Audio | SI-SDR (dB) | STOI | PESQ-WB |
| --- | ---: | ---: | ---: |
| Noisy input | 4.779 | 0.9401 | 2.059 |
| BigCodec CPU reconstruction | -1.025 | 0.8831 | 1.651 |
| BigCodec FPGA BF16 reconstruction | -1.505 | 0.8839 | 1.633 |
| DPDFNet 16-kHz FPGA enhancement | 18.112 | 0.9360 | 2.756 |

DPDFNet improves SI-SDR and PESQ on this recording; STOI is slightly below the noisy input. This is one paired test, not a general quality benchmark. BigCodec's CPU and FPGA reconstructions both score below the noisy input here.

The BigCodec FPGA file comes from build `0x40519e0a`. The DPDFNet file is a previously recorded result from build `0xdf0749de`; this check does not rerun hardware. [Exact input/output hashes and scores](bus_clean_reference_quality.json) identify every file. The clean reference remains in the ignored dataset cache identified by that manifest.

Listen to the same 23.7-second input:

- [Noisy bus input](../../../../dpdfnet/validation/20260914_noisy20s/noisy/bus_noisy.wav)
- [BigCodec CPU reconstruction](../../20260914_noisy20s/cpu/bus.wav)
- [BigCodec FPGA reconstruction](../decoder/fpga_bf16/bus.wav)
- [DPDFNet FPGA denoised output](../../../../dpdfnet/validation/20260914_noisy20s/fpga16k/bus_fpga_16khz.wav)

The [newer DPDFNet precision report](../../../../dpdfnet/validation/20260914_bf16/README.md) also includes 20-second bus and café examples on build `0x40519e0a`.

Recompute the quality table from the repository root, with the paired clean cache and metric dependencies installed:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python \
  models/bigcodec/validation/20260915_accuracy/filtering_check/reproduce_quality.py
```
