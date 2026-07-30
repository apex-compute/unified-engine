# Llama 3.2 1B Hardware Benchmark

Benchmarked with the default prompt and `decode-kernel=streaming`. Each result
is from a fresh `make clean` followed by one hardware run. The RK-256b and
Kintex-7 results were collected at `f1796e6`; RK-512b was collected from the
updated working tree based on `f17b6e1`. All ten configurations decoded the
expected answer.

<table>
  <thead>
    <tr>
      <th rowspan="3">Metric</th>
      <th colspan="4">RK-256b (xdma0, 3.0000 ns)</th>
      <th colspan="2">RK-512b (xdma0, 3.0000 ns)</th>
      <th colspan="4">Kintex-7 (xdma1, 5.0422 ns)</th>
    </tr>
    <tr>
      <th colspan="2">IF4</th>
      <th colspan="2">IF8</th>
      <th>IF4</th>
      <th>IF8</th>
      <th colspan="2">IF4</th>
      <th colspan="2">IF8</th>
    </tr>
    <tr>
      <th>Streaming</th>
      <th>Matmatmul</th>
      <th>Streaming</th>
      <th>Matmatmul</th>
      <th>Streaming</th>
      <th>Streaming</th>
      <th>Streaming</th>
      <th>Matmatmul</th>
      <th>Streaming</th>
      <th>Matmatmul</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Hardware frequency</td>
      <td colspan="4">333 MHz</td>
      <td colspan="2">333 MHz</td>
      <td colspan="4">198.33 MHz</td>
    </tr>
    <tr>
      <td>DDR configuration</td>
      <td colspan="4">DDR4 @ 1333 MHz, 32-bit</td>
      <td colspan="2">DDR4 @ 1333 MHz, 32-bit</td>
      <td colspan="4">DDR3 @ 533.33 MHz, 72-bit</td>
    </tr>
    <tr>
      <td>DRAM test speed</td>
      <td colspan="4">9.5 GB/s read, 8.3 GB/s write</td>
      <td colspan="2">9.5 GB/s read, 8.3 GB/s write</td>
      <td colspan="4">6.1 GB/s read, 5.8 GB/s write</td>
    </tr>
    <tr>
      <td>Prefill tokens</td>
      <td>44</td><td>44</td><td>44</td><td>44</td>
      <td>44</td><td>44</td>
      <td>44</td><td>44</td><td>44</td><td>44</td>
    </tr>
    <tr>
      <td>TTFT / prefill HW time (ms)</td>
      <td>2,330.1</td><td>2,290.2</td><td>4,494.0</td><td>2,353.0</td>
      <td>2,330.8</td><td>4,494.3</td>
      <td>3,663.9</td><td>3,864.5</td><td>7,037.2</td><td>3,962.4</td>
    </tr>
    <tr>
      <td>Prefill speed (tok/s)</td>
      <td>18.88</td><td>19.21</td><td>9.79</td><td>18.70</td>
      <td>18.88</td><td>9.79</td>
      <td>12.01</td><td>11.39</td><td>6.26</td><td>11.11</td>
    </tr>
    <tr>
      <td>Prefill throughput (GFLOPS)</td>
      <td>38.07</td><td>38.73</td><td>19.74</td><td>37.70</td>
      <td>38.05</td><td>19.74</td>
      <td>24.21</td><td>22.95</td><td>12.60</td><td>22.38</td>
    </tr>
    <tr>
      <td>Peak decode speed, first token (tok/s)</td>
      <td>13.33</td><td>13.33</td><td>7.27</td><td>7.27</td>
      <td>13.32</td><td>7.27</td>
      <td>8.45</td><td>8.45</td><td>4.64</td><td>4.64</td>
    </tr>
    <tr>
      <td>Average decode speed (tok/s)</td>
      <td>12.93</td><td>12.93</td><td>7.13</td><td>7.13</td>
      <td>12.90</td><td>7.16</td>
      <td>8.24</td><td>8.24</td><td>4.57</td><td>4.57</td>
    </tr>
    <tr>
      <td>Decode throughput (GFLOPS)</td>
      <td>34.10</td><td>34.10</td><td>18.77</td><td>18.77</td>
      <td>34.09</td><td>18.77</td>
      <td>21.60</td><td>21.61</td><td>11.96</td><td>11.96</td>
    </tr>
  </tbody>
</table>

## Commands

RK-256b:

```bash
python models/llama3.2_1b/llama3.2_1b_test.py --device rk --prefill-kernel streaming
python models/llama3.2_1b/llama3.2_1b_test.py --device rk --prefill-kernel matmatmul
python models/llama3.2_1b/llama3.2_1b_IF8.py --device rk --prefill-kernel streaming
python models/llama3.2_1b/llama3.2_1b_IF8.py --device rk --prefill-kernel matmatmul
```

RK-512b (streaming prefill only; matmatmul does not apply):

```bash
python models/llama3.2_1b/llama3.2_1b_test.py --device rk --prefill-kernel streaming
python models/llama3.2_1b/llama3.2_1b_IF8.py --device rk --prefill-kernel streaming
```

Kintex-7:

```bash
python models/llama3.2_1b/llama3.2_1b_test.py --device kintex7 --dev xdma1 --prefill-kernel streaming
python models/llama3.2_1b/llama3.2_1b_test.py --device kintex7 --dev xdma1 --prefill-kernel matmatmul
python models/llama3.2_1b/llama3.2_1b_IF8.py --device kintex7 --dev xdma1 --prefill-kernel streaming
python models/llama3.2_1b/llama3.2_1b_IF8.py --device kintex7 --dev xdma1 --prefill-kernel matmatmul
```

`TTFT` here is the reported hardware prefill execution time. It excludes model
loading, weight transfer, compilation, and host-side setup. Because the decoder
kernel remains streaming in every run, changing only the prefill kernel does not
materially change decode performance.
