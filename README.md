# vizy: One-line tensor visualization for PyTorch and NumPy

**Stop juggling tensor formats. Just visualize.**

Display or save any NumPy array or PyTorch tensor (supports 2D, 3D, 4D shapes) with a single line:

```python
import vizy

vizy.plot(tensor)               # shows image or grid
vizy.save("image.png", tensor)  # saves to file
vizy.save(tensor)               # saves to temp file and prints path
vizy.summary(tensor)            # prints info like res, dtype, device, range, etc.
```

Let's say you have a PyTorch `tensor` with shape `(BS, 3, H, W)`. Instead of

```python
plt.imshow(tensor.cpu().numpy()[0].transpose(1, 2, 0))
plt.imshow(tensor.cpu().numpy()[1].transpose(1, 2, 0))
...
```

You can just do:

```python
vizy.plot(tensor)
```

Or if you are in an ssh session, you can just do:

```python
vizy.save(tensor)
```

It will automatically save the tensor to a temporary file and print the path, so you can scp it to your local machine and visualize it.

## Technical Details

**Note**: This library uses [matplotlib](https://github.com/matplotlib/matplotlib) for visualization under the hood. It handles tensor format detection, conversion, and grid layout automatically, but the actual plotting is done via matplotlib.

## Installation

```bash
pip install vizy
```
