# vizy

**Lightweight tensor visualizer for PyTorch and NumPy**

Display or save any tensor with a single line:

```python
import vizy

vizy.plot(tensor)               # shows image or grid
vizy.save("image.png", tensor)  # saves to file
vizy.save(tensor)               # saves to temp file and prints path
```

Supports tensors or arrays with shape like:
- `(H, W)`
- `(C, H, W)` → auto-converted to `(H, W, C)`
- `(1, 1, H, W)` → squeezed
- `(B, C, H, W)` → shown as grid

## Installation

```bash
pip install vizy
```
