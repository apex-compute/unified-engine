# Audio sources

The eight tests use original VoiceBank-DEMAND mixtures of recorded speech and
environmental noise: bus, café, office and public square.

Speech dataset: Cassia Valentini-Botinhao (2017), University of Edinburgh/CSTR,
*Noisy speech database for training speech enhancement algorithms and TTS
models, 2016*, [DOI 10.7488/ds/2117](https://doi.org/10.7488/ds/2117),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Environmental recordings: Joachim Thiemann, Nobutaka Ito and Emmanuel Vincent
(2013), [DEMAND](https://zenodo.org/records/1227121),
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).

The four retained tests contain the original 2.5, 7.5, 12.5 and 17.5 dB nominal
SNR conditions. Each repeats its shortest complete utterance once to exceed
20 seconds. The four added `*_low_snr` tests use only 2.5 dB utterances absent
from the original pinned subset. Selection is fixed before model evaluation:
choose the fewest distinct eligible clips reaching 20–24 seconds, include both
speakers where available, minimize distance to 21.5 seconds, and break ties by
sorted utterance IDs. Added tests contain no repeated clips.

All clips are resampled directly from 48 kHz to mono 16 kHz FLOAT WAV and joined
with 250 ms zero gaps. Utterances remain complete, with no gain normalization,
crossfade or new noise mixing. Paired clean references use the same operations.
SNR labels come from the original condition log and are not recomputed for
the concatenated tests.

Sources are extracted from the official [clean archive](https://datashare.ed.ac.uk/server/api/core/bitstreams/dec213d3-bf57-4777-9663-c24bdce92d5e/content)
and [noisy archive](https://datashare.ed.ac.uk/server/api/core/bitstreams/13c1bfbf-14a6-41db-9b41-8f7310f01ad5/content),
with conditions from the [original metadata](https://datashare.ed.ac.uk/server/api/core/bitstreams/11185dc8-9cf1-405b-b858-35bd6a04aedd/content).
Archive member sizes/CRC32 and individual source SHA-256 hashes are checked.

The [input manifest](input_manifest.json), [additional source records](additional_sources.json)
and [frozen selection](additional_source_selection.json) preserve provenance,
segment boundaries, hashes and sample counts. Historical source records remain unchanged; current tests are listed
explicitly in that manifest. The [original subset](../../noisy_test_cases.json)
and [noisy-test documentation](../../NOISY_TESTS.md) provide earlier provenance.
