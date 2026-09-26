# Sample data

`btc_lob_sample_100k.csv.gz` contains the first 100,000 snapshots (2023-01-09 22:17 to
2023-01-10 05:15 UTC, about 7 hours) of the Kaggle dataset
[Bitcoin Limit Order Book (LOB) Data](https://www.kaggle.com/datasets/siavashraz/bitcoin-perpetualbtcusdtp-limit-order-book-data)
(BTCUSDT perpetual, 10 levels per side, ~250 ms sampling). The file is unchanged apart
from being truncated and gzipped. The full dataset (3.73M rows, 1.2 GB) is not included.

Same 43-column schema as the full file: row index, epoch ms, UTC datetime, then 10
(price, volume) bid levels best first, then 10 ask levels best first.

```bash
python scripts/prepare_data.py --csv sample_data/btc_lob_sample_100k.csv.gz --cache data/sample_real/cache
python scripts/run_obi_sweep.py --cache data/sample_real/cache --out results/sample --figures results/sample
```

## Dataset license

The dataset is released on Kaggle under the MIT License; its notice is reproduced below.

```
Copyright (c) 2013 Mark Otto.

Copyright (c) 2017 Andrew Fong.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
