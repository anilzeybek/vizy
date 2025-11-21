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

**Example output:**

![Example notebook showing vizy.plot() with different tensor formats](assets/example.png)

Or if you are in an ssh session, you can just do:

```python
vizy.save(tensor)
```

It will automatically save the tensor to a temporary file and print the path, so you can scp it to your local machine and visualize it.

## Technical Details

**Note**: This library uses [PIL/Pillow](https://github.com/python-pillow/Pillow) for image visualization and saving. This library handles tensor format detection, conversion, and grid layout, but the actual plotting is done via PIL.

## Installation

```bash
pip install vizy
```

<a href="https://www.buymeacoffee.com/anilz" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>
