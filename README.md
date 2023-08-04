# xrsignal

<!-- SPHINX-START -->

- The goal here is to implement scipy.signal functions using map_blocks on an as needed basis
- In general, I'm going to default to implementing function only within dask blocks.
    - so for instance filtering would only be within a chunk, and the boundaries of the chunk would have innacurate values.
    - There are likely ways to address more rigoriously that might be explored in the future.

## Installation

- for now, while xrsignal is under development, xrsignal isn't available on pypi or conda
- to install, please clone the repository and add the directory that the package is installed to your python path

### In a conda environment:
```
cd /path/to/code/dir/
git clone https://github.com/John-Ragland/xrsignal.git

conda develop /path/to/code/dir/
```