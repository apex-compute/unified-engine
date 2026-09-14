# Audio sources and transformations

These five environment sequences use the original VoiceBank-DEMAND test-set
mixtures of recorded clean speech and recorded environmental noise. They are
not live recordings of a speaker in those environments.

Cassia Valentini-Botinhao (2017), University of Edinburgh/CSTR, *Noisy speech
database for training speech enhancement algorithms and TTS models, 2016*.
The saved source metadata identifies [DOI 10.7488/ds/2117](https://doi.org/10.7488/ds/2117)
and the [University of Edinburgh dataset](https://datashare.ed.ac.uk/handle/10283/2791),
with [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The dataset's [license source document](https://datashare.ed.ac.uk/server/api/core/bitstreams/7e7bc32f-e94a-436a-8e97-416a972c7e1a/content)
is recorded in the source provenance.

The environmental recordings originate from Joachim Thiemann, Nobutaka Ito
and Emmanuel Vincent (2013), [DEMAND: Diverse Environments Multichannel
Acoustic Noise Database](https://zenodo.org/records/1227121), identified in the
saved metadata as [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
The environments are public transit bus (TBUS), café terrace (SCAFE), living
room (DLIVING), small office (OOFFICE), and public town square (SPSQUARE).

Each sequence includes all four available nominal SNR conditions for its
environment: 2.5, 7.5, 12.5 and 17.5 dB, in ascending order. These labels come
from the original dataset log; they are not remeasured composite SNRs.
The shortest complete utterance is repeated once, adjacent to its SNR entry,
to bring each sequence above 20 seconds. There are 250 ms zero-silence gaps
between utterances. Each original mono 48 kHz utterance is resampled directly
to 16 kHz once, then concatenated. The paired clean references receive the
same sequence of operations. No new noise is mixed, and no gain normalization,
crossfade, or utterance truncation is applied. Outputs are mono FLOAT WAVs.

The original files were obtained from the dataset's [clean test archive](https://datashare.ed.ac.uk/server/api/core/bitstreams/dec213d3-bf57-4777-9663-c24bdce92d5e/content)
and [noisy test archive](https://datashare.ed.ac.uk/server/api/core/bitstreams/13c1bfbf-14a6-41db-9b41-8f7310f01ad5/content).
Source provenance records archive member names, CRC32 values, byte lengths,
and per-WAV SHA-256 hashes. Conditions are taken from `log_testset.txt` in the
[original condition metadata](https://datashare.ed.ac.uk/server/api/core/bitstreams/11185dc8-9cf1-405b-b858-35bd6a04aedd/content).

The [input manifest](input_manifest.json) records exact sequence boundaries,
repeated utterances, source and composite hashes, and sample counts.
The [pinned source selection](../../noisy_test_cases.json) and
[noisy-test documentation](../../NOISY_TESTS.md) provide the original subset
selection and provenance details. The source records bound by this sequence
manifest have these SHA-256 hashes:

| Record | SHA-256 |
|---|---|
| Downloaded source `cases.json` | `81655a2033c2cc8fad20e461861a8ced41de927f1cef8c0fd05787ceced16d4c` |
| Downloaded source `provenance.json` | `da2ec363b41431af3aaa1446f4d05eb3232fb8b36e8dc0f769aff9c6ffd11a3d` |
