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

data = data_loader.DataLoader('RadiusEarthData')
model = fit_models.Linear()

fit = fitting.Fitting(model, data.x, data.y, data.y_error)


fig, ax = plt.subplots(1, 1, figsize=(16,9))
residuals_fig, residuals_ax = plt.subplots(1, 1, figsize=(16,9))


fit.scatter_plot_data_and_fit(ax)

fit.plot_residuals(residuals_ax)

fig.show()
residuals_fig.show()