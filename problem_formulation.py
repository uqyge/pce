# %%
import numpy

coordinates = numpy.linspace(0, 10, 1000)


def model_solver(parameters):

    alpha, beta = parameters
    return alpha*numpy.e**-(coordinates*beta)
# %%
from matplotlib import pyplot

pyplot.plot(coordinates, model_solver([1.5, 0.1]))
pyplot.plot(coordinates, model_solver([1.7, 0.2]))
pyplot.plot(coordinates, model_solver([1.7, 0.1]))

pyplot.show()
# %%
import chaospy

alpha = chaospy.Normal(1.5, 0.2)
beta = chaospy.Uniform(0.1, 0.2)
joint = chaospy.J(alpha, beta)

# %%
grid = numpy.mgrid[1:2:100j, 0.1:0.2:100j]
contour = pyplot.contourf(grid[0], grid[1], joint.pdf(grid), 50)

pyplot.scatter(*joint.sample(50, seed=1234))

pyplot.show()
# %%
parameter_samples = joint.sample(10000, seed=1234)
model_evaluations = numpy.array([model_solver(sample)
                                for sample in parameter_samples.T])
# %%
pyplot.plot(coordinates, model_evaluations[:1000].T, alpha=0.03)
pyplot.show()
# %%
_t = coordinates[1:]

true_mean = numpy.hstack([
    1.5, 15*(numpy.e**(-0.1*_t)-numpy.e**(-0.2*_t))/_t])
true_variance = numpy.hstack([
    2.29, 11.45*(numpy.e**(-0.2*_t)-numpy.e**(-0.4*_t))/_t])-true_mean**2
# %%
std = numpy.sqrt(true_variance)
pyplot.fill_between(coordinates, true_mean-2*std, true_mean+2*std, alpha=0.4)
pyplot.plot(coordinates, true_mean)
pyplot.show()
# %%
def error_in_mean(predicted_mean, true_mean=true_mean):
    return numpy.mean(numpy.abs(predicted_mean-true_mean))

def error_in_variance(predicted_variance,
                      true_variance=true_variance):
    return numpy.mean(numpy.abs(predicted_variance-true_variance))

# %%
(error_in_mean(numpy.mean(model_evaluations[:100], 0)),
 error_in_variance(numpy.var(model_evaluations[:100], 0)))
# %%
indices = numpy.arange(100, 10001, 100, dtype=int)
eps_mean = [error_in_mean(numpy.mean(model_evaluations[:idx], 0))
            for idx in indices]
eps_variance = [error_in_variance(numpy.var(model_evaluations[:idx], 0))
                for idx in indices]
# %%
pyplot.semilogy(indices, eps_mean, "-", label="mean")
pyplot.semilogy(indices, eps_variance, "--", label="variance")
pyplot.legend()
pyplot.show()
# %%
