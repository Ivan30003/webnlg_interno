# Utils
This directory includes utility scripts to help you process the WebNLG data.

## Reading an XML Dataset
You can quickly parse a WebNLG XML file with the `benchmark_reader.py` script. The original implementation is from [this repo](https://gitlab.com/webnlg/corpus-reader/-/tree/master).

```python
from benchmark_reader import Benchmark
from benchmark_reader import select_files

b = Benchmark()
files = select_files("data/public/br_train.xml")
b.fill_benchmark(files)
```

The `Benchmark` class has utility functions for translating between different data structures and file formats.

```python
corpus = b.to_dict() # Benchmark -> Dictionary

b.b2json("some_path/", "some_file.json") # save in JSON format

b.b2xml("some_path/", "some_file.xml") # re-save in XML format
```
