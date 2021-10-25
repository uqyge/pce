#%%
from chaospy import expansion
from mpl_toolkits import mplot3d
import numpy as np
import chaospy
import matplotlib.pyplot as plt

# %%
coordinates = np.linspace(0, 10, 1000)


def model_solver(parameters, coordinates=coordinates):
    alpha, beta = parameters
    return alpha * np.e ** (-coordinates * beta)


# %%
for params in [(1.3, 0.13), (1.7, 0.17), (1.1, 0.19), (1.9, 0.11)]:
    print(params)
    plt.plot(coordinates, model_solver(params))
plt.show()

# %%
alpha = chaospy.Normal(1.5, 0.2)
beta = chaospy.Uniform(0.1, 0.2)
joint = chaospy.J(alpha, beta)

# %%
grid = np.mgrid[0.9:2.1:100j, 0.09:0.21:100j]
plt.contourf(grid[0], grid[1], joint.pdf(grid), 30)
plt.scatter(*joint.sample(100, rule="sobol"))
plt.show()
# %%
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_wireframe(grid[0], grid[1], joint.pdf(grid))
plt.show()
# %%
import plotly.express as px
import pandas as pd

# %%
df = pd.DataFrame(
    joint.sample(1000, rule="sobol").T,
    columns=["alpha", "beta"],
)

fig = px.scatter(
    df,
    x="alpha",
    y="beta",
    marginal_x="histogram",
    marginal_y="histogram",
)
fig.show()

fig = px.density_heatmap(
    df,
    x="alpha",
    y="beta",
    marginal_x="histogram",
    marginal_y="histogram",
)
fig.show()


# %%
expansion = chaospy.generate_expansion(8, joint)
len(expansion)
# %%
samples = joint.sample(1_000, rule="sobol")
evaluations = np.array([model_solver(sample) for sample in samples.T])
plt.plot(coordinates, evaluations[::10].T, alpha=0.2)
plt.show()
# %%
approx_solver = chaospy.fit_regression(expansion, samples, evaluations)

# %%
expected = chaospy.E(approx_solver, joint)
deviation = chaospy.Std(approx_solver, joint)
# %%
plt.fill_between(
    coordinates, expected - 2 * deviation, expected + 2 * deviation, alpha=0.2
)
plt.plot(coordinates, expected)
plt.show()
# %%
a = ["1", "2", "3", "4"]
# %%
print(a)
# %%
print(*a)
# %%
[*a]
# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
x = np.linspace(1, 10, 10)
y = x * 2
c = [x, y]
# %%
plt.plot(x, y)


# %%
plt.plot(*c)
# %%
