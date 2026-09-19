# Audio attribution

[audio/input.wav](audio/input.wav) is the unmodified `noisy_testset_wav/p232_007.wav` recording from the [VoiceBank-DEMAND test set](https://datashare.ed.ac.uk/handle/10283/2791). It contains speaker `p232` with café noise at 12.5 dB nominal SNR, identified by the original condition log. SHA256: `4d36133cdfd72c5de9379bae4139fc534ae5498a7b788d6c1f346feeeffec211`.

Attribution: Cassia Valentini-Botinhao (2017), *Noisy speech database for training speech enhancement algorithms and TTS models, 2016*, University of Edinburgh, School of Informatics, Centre for Speech Technology Research. [DOI: 10.7488/ds/2117](https://doi.org/10.7488/ds/2117).

The paired dataset distributes this recording under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The included [license text](p232_007.LICENSE.txt) is copied from the dataset's original license. Environmental noise originates from [DEMAND](https://zenodo.org/records/1227121), Joachim Thiemann, Nobutaka Ito and Emmanuel Vincent (2013), licensed CC BY-SA 3.0.

The seven other WAVs in `audio/` are codec reconstructions of this recording: official CPU output, three measured FPGA outputs and three official CPU decodes of the corresponding FPGA token sequences. They were resampled through the 16 kHz model and restored to the original 48 kHz rate and sample count. They are modified derivatives, not original dataset recordings.

BigCodec source: Xin Detai (2024), [upstream commit 09845ab1f5bc7a3d1589c16de820ef15bc2afefe](https://github.com/Aria-K-Alethia/BigCodec/tree/09845ab1f5bc7a3d1589c16de820ef15bc2afefe), MIT. The separately downloaded [checkpoint](https://huggingface.co/Alethia/BigCodec/tree/c6548832eda95a09dcc3485d9b8ef097ce15f387) is identified as CC-BY-SA-4.0 by its model card. Neither model source nor checkpoint is included in this evidence folder; checkpoint provenance and hash are in [results.json](results.json).
