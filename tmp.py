#%%
import chaospy
import matplotlib.pyplot as plt

# %%
normal = chaospy.Normal(mu=2, sigma=2)
normal
# %%
samples = normal.sample(4, seed=1234)
samples
# %%


# %%
plt.hist(normal.sample(10_000, seed=1234), 30)
plt.show()
# %%
normal.pdf([-2, 0, 2])

# %%
import numpy as np

# %%
q_loc = np.linspace(-4, 8, 200)
plt.plot(q_loc, normal.pdf(q_loc))
plt.show()

# %%
normal.cdf([-2, 0, 2])
# %%
plt.plot(q_loc, normal.cdf(q_loc))
plt.show()
# %%
normal.mom([0, 1, 2])
# %%
chaospy.approximate_moment(normal, [2])
# %%
normal_trunc = chaospy.Trunc(normal, upper=4)
# %%
plt.plot(q_loc, normal_trunc.pdf(q_loc))
plt.plot(q_loc, normal.pdf(q_loc))
plt.show()
# %%
chaospy.approximate_moment(normal_trunc, [0])
# %%
from problem_formulation import joint

# %%
gauss_quads = [
    chaospy.generate_quadrature(order, joint, rule="gaussian") for order in range(1, 8)
]

# %%
sparse_grid = [
    chaospy.generate_quadrature(
        order, joint, rule=["genz_keister_24", "clenshaw_curtis"], sparse=True
    )
    for order in range(1, 5)
]

# %%
for order in range(5):
    # order = 1
    plt.figure()
    plt.subplot(121)
    nodes, weights = gauss_quads[order]
    plt.scatter(*nodes, s=weights * 1000)
    nodes, weights = sparse_grid[order]
    plt.subplot(122)
    plt.scatter(*nodes, s=weights * 1000)
# %%
