"""Run native 8 kHz DPDFNet2 with BF16 dense convolution weights.

Other learned weights and activations also use BF16.
"""

from pathlib import Path

from dpdfnet8khz_run_from_bin import main as run_main


HERE = Path(__file__).resolve().parent


def main(argv=None):
    run_main(
        argv,
        default_bin=HERE / "dpdfnet8khz_bin/dpdfnet2_8khz-bf16-andromeda.bin",
        expected_precision="bf16",
    )


if __name__ == "__main__":
    main()
