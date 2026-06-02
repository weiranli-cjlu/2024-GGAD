## Requirements

To install requirements:

```bash
uv venv -p 3.12
uv pip install torch==2.4.0 scikit-learn --torch-backend=cpu
uv pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/repo.html
```
