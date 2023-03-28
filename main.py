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
known_radius = 6378100 # m
theoretical_slope = -2 * g / known_radius

data = data_loader.DataLoader('RadiusEarthData')
model = fit_models.Proportional()

units_for_parameters = (r's$^{-2}$')

fit = fitting.Fitting(model, x=data.x, y_measured=data.y, x_error=data.x_error, y_error=data.y_error, units_for_parameters=units_for_parameters)


fig, ax = plt.subplots(1, 1, figsize=(16,9))
residuals_fig, residuals_ax = plt.subplots(1, 1, figsize=(16,9))

x_for_theoretical_plot = np.linspace(min(data.x), max(data.x), 10000)
y_for_theoretical_plot = theoretical_slope * x_for_theoretical_plot

ax.plot(x_for_theoretical_plot, y_for_theoretical_plot, label='theoretical line', color='k')

fit.scatter_plot_data_and_fit(ax)
fit.plot_residuals(residuals_ax)

ax.set_title(r'$\Delta g$ vs. $\Delta r$ for Sodin Gravimeter readings in Burton Tower')
ax.set_ylabel(r'$\Delta g$ / m s$^{-2}$')
ax.set_xlabel(r'$\Delta r$ / m')

residuals_ax.set_title(r'Residuals for proportional fit to $\Delta g$ vs. $\Delta r$')
residuals_ax.set_ylabel(r'residuals / m s$^{-2}$')
residuals_ax.set_xlabel(r'$\Delta r$ / m')

if data.all_floors:
        fig.savefig('plots/fit_plot_all_floors.png')
        residuals_fig.savefig('plots/residuals_plot_all_floors.png')
else:
        fig.savefig('plots/fit_plot.png')
        residuals_fig.savefig('plots/residuals_plot.png')


# fig.show()
# residuals_fig.show()

slope = fit.popt[0]
slope_error = fit.parameter_errors[0]

def R_from_slope(slope):
        result = -2 * g / slope
        return result

def R_from_slope_error(slope, slope_error):
        error = 2 * g / slope**2 * slope_error
        return error

R_best_guess = R_from_slope(slope)
R_error = R_from_slope_error(slope, slope_error)

print('measured radius is ' + Output.print_with_uncertainty(R_best_guess, R_error))
print(R_best_guess / known_radius, 'of the known value')