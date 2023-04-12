# xrsignal

- The goal here is to implement scipy.signal functions using map_blocks on an as needed basis
- In general, I'm going to default to implementing function only within dask blocks.
    - so for instance filtering would only be within a chunk, and the boundaries of the chunk would have innacurate values.
    - There are likely ways to address more rigoriously that might be explored in the future.