import numpy as np

x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)

u = np.sin(np.pi * X) * np.cos(np.pi * Y)
v = -np.cos(np.pi * X) * np.sin(np.pi * Y)
p = 101325 + 50 * np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)

coords = np.column_stack([X.flatten(), Y.flatten()])

np.savez("examples/test_vort.npz", coords=coords, u=u.flatten(), v=v.flatten(), p=p.flatten())

print("test_vort.npz created successfully")
