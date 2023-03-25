import matplotlib.pyplot as plt
from matplotlib import rc
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 16}
rc('font', **font)
import numpy as np

import fitting_and_analysis
Output = fitting_and_analysis.Output()
import fitting
import data_loader
import fit_models

g = 9.804253 # m s^-2
known_radius = 6378100
theoretical_slope = -2 * g / known_radius

data = data_loader.DataLoader('RadiusEarthData')
model = fit_models.Proportional()

fit = fitting.Fitting(model, data.x, data.y, data.y_error)


fig, ax = plt.subplots(1, 1, figsize=(16,9))
residuals_fig, residuals_ax = plt.subplots(1, 1, figsize=(16,9))

x_for_theoretical_plot = np.linspace(min(data.x), max(data.x), 10000)
y_for_theoretical_plot = theoretical_slope * x_for_theoretical_plot

ax.plot(x_for_theoretical_plot, y_for_theoretical_plot, label='theoretical line', color='k')

fit.scatter_plot_data_and_fit(ax)
fit.plot_residuals(residuals_ax)


fig.savefig('plots/fit_plot.png')
residuals_fig.savefig('plots/residuals_plot.png')



# fig.show()
# residuals_fig.show()

slope = fit.popt[0]
slope_error = fit.parameter_errors[0]

print(slope, slope_error)

def R_from_slope(slope):
        result = -2 * g / slope
        return result

def R_from_slope_error(slope, slope_error):
        error = 2 * g / slope**2 * slope_error
        return error

R_best_guess = R_from_slope(slope)
R_error = R_from_slope_error(slope, slope_error)

print(R_best_guess, R_error)

print('measured radius is ' + Output.print_with_uncertainty(R_best_guess, R_error))
print(R_best_guess / known_radius)