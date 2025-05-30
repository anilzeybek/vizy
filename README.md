# vizy

**Lightweight tensor visualizer for PyTorch and NumPy**

Display or save any NumPy array or PyTorch tensor with a single line with ease:

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


## Installation

```bash
pip install vizy
```
