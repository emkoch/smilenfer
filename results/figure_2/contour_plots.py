"""
Functions to plot probability densities of variants at different 
allele frequencies, conditional on their effect sizes and ascertainment.

We will use functions from smilenfer.simulation to allele frequencies densities
on a grid of effect sizes and the apply ascertainment to these densities
and normalize them to get the conditional densities.

Will also include some example plots of the above.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import smilenfer.simulation as sim
from smilenfer.statistics import trad_x_set

## Helper functions
def calculate_normalization_factors(density_grid, x_set, discovery_x, n_x=2000):
    normalization_factors = np.zeros_like(discovery_x)
    for ii, xx in enumerate(discovery_x):
        restricted_x_set = trad_x_set(xx, n_x)
        restricted_density_set = np.interp(restricted_x_set, x_set, density_grid[:,ii])
        normalization_factors[ii] = np.trapz(restricted_density_set, restricted_x_set)
    return normalization_factors

def normalize_grid(density_grid, x_set, discovery_x, normalization_factors):
    result = density_grid / normalization_factors
    # Set densities outside the ascertainment window to NaN
    result[x_set[:,None] < discovery_x[None,:]] = np.nan
    result[x_set[:,None] > (1-discovery_x[None,:])] = np.nan
    return result

# At present consider the following models:
# - neutral
# - directional
# - stabilizing
# - pleiotropic stabilizing

def conditional_contour_plot(ax, x_set, beta_set, density_grid, cdf_min=0.025, cdf_max=0.975, 
                             n_levels=9, density_min=None, density_max=None, cmap="jet", xlabel="MAF", ylabel=r"$\beta$",
                             add_colorbar=True, levels=None, use_contourf=True, linewidths=1):
    # chosoe various cutoff values to make sure the plot looks good
    density_tmp = np.copy(density_grid)
    # set NaNs to 0
    density_tmp[np.isnan(density_tmp)] = 0
    # fold the density grid
    density_tmp += np.flip(density_tmp, axis=0)
    density_tmp = density_tmp[x_set <= 0.5, :]
    maf_set = x_set[x_set <= 0.5]
    maf_diff = np.diff(maf_set)
    density_midpoints = (density_tmp[1:,:] + density_tmp[:-1,:])/2
    x_cdf_grid = np.cumsum(density_midpoints * maf_diff[:,None], axis=0)
    # append a row of zeros to the start of the grid
    x_cdf_grid = np.vstack((np.zeros_like(x_cdf_grid[0,:]), x_cdf_grid))
    # set all zeros to NaNs
    # x_cdf_grid[x_cdf_grid == 0] = np.nan
    # calculate the CDF at each x value, taking account of the fact that the x grid is uneven
    # contour = ax.contour(maf_set, beta_set, x_cdf_grid.T, levels=np.linspace(0.1, 0.99, 9), cmap='cividis')
    # ax.set_ylabel(r'$\beta$')
    # ax.set_xlabel('MAF')
    # Add the colorbar for the contour plot
    # plt.colorbar(contour)

    if density_max is None and density_min is None:
        # Get the x values at which the CDF is 0.025 and 0.975
        x_lower = np.zeros_like(beta_set)
        x_upper = np.zeros_like(beta_set)
        for ii, beta in enumerate(beta_set):
            x_lower[ii] = maf_set[np.argmin(np.abs(x_cdf_grid[:,ii] - cdf_min))]
            x_upper[ii] = maf_set[np.argmin(np.abs(x_cdf_grid[:,ii] - cdf_max))]

        # Get the density values for the lower and upper bounds from density_grid
        density_lower = np.zeros_like(beta_set)
        density_upper = np.zeros_like(beta_set)
        for ii, beta in enumerate(beta_set):
            density_lower[ii] = density_grid[x_set == x_lower[ii], ii]
            density_upper[ii] = density_grid[x_set == x_upper[ii], ii]

        # Get minimum and maximum density values
        # print(np.nanmin(density_lower), np.nanmax(density_upper))
        density_max = np.log2(np.nanmax(density_lower))
        density_min = np.log2(np.nanmin(density_upper))

    # copy the density grid
    density_plot = np.copy(density_grid)
    if levels is None:
        pass
        # set everything above density_max to density_max
        density_plot[density_plot > 2**density_max] = 2**density_max 
        # set everything below density_min to density_min
        density_plot[density_plot < 2**density_min] = 2**density_min
    else:
        print("Using custom levels")
        density_plot[density_plot > 2**levels[-1]] = 2**levels[-1]
        density_plot[density_plot < 2**levels[0]] = 2**levels[0]

    if levels is None:
        if n_levels is None:
            levels = np.arange(density_min, density_max+1, 1)
        else:
            levels = np.linspace(density_min, density_max, n_levels)
        print(levels)
        if use_contourf:
            contour = ax.contourf(x_set, beta_set, np.log2(density_plot).T, levels=levels, cmap=cmap)
        else:
            contour = ax.contour(x_set, beta_set, np.log2(density_plot).T, 
                                 levels=levels, cmap=cmap, linewidths=linewidths)
        print(contour.levels)
    else:
        if use_contourf:
            contour = ax.contourf(x_set, beta_set, np.log2(density_plot).T, levels=levels, cmap=cmap)
        else:
            contour = ax.contour(x_set, beta_set, np.log2(density_plot).T, levels=levels, cmap=cmap, linewidths=linewidths)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    # Add the colorbar for the contour plot
    if add_colorbar:
        plt.colorbar(contour)

    return contour, density_min, density_max
    

# A function to get the conditional density grid for the neutral model
# TODO: betas not used, remove
def neutral_conditional_densities(beta, x_set, discovery_x, pile=None, n_x=2000):
    """
    Get the conditional densities of variants at different allele frequencies,
    conditional on their effect sizes and ascertainment, for the neutral model.
    
    Parameters
    ----------
    beta : array-like
        Grid of effect sizes to plot densities for.
    x_set : array-like
        Set of allele frequencies to plot densities for.
    discovery_x : array-like
        Set of minumum minor allele frequencies to normalize densities by.
    pile : simulation.Pile, optional
        Pile dict to use to get the densities. If not provided, use equilibrium densities.
    """
    if pile is None:
        density_set = sim.sfs_neut_params(x_set, 1)
        density_set += sim.sfs_neut_params(1-x_set, 1)
    else:
        S_0_ii = np.where(pile["s_set"] == 0)[0][0]
        S_ud_0_ii = np.where(pile["s_ud_set"] == 0)[0][0]
        sfs_neut = pile["sfs_grid"][S_0_ii, S_ud_0_ii]
        density_set = np.interp(x_set, pile["interp_x"], sfs_neut)
        density_set += np.interp(1-x_set, pile["interp_x"], sfs_neut)
    density_grid = density_set[:,None] * np.ones_like(beta)
    # Calculate normalization factors for each effect size
    normalization_factors = calculate_normalization_factors(density_grid, x_set, discovery_x, n_x)
    density_grid = normalize_grid(density_grid, x_set, discovery_x, normalization_factors) 
    return density_grid

# A function to get the conditional density grid for the directional model
def directional_conditional_densities(beta, x_set, discovery_x, I1, Ne, pile=None, n_x=2000):
    S_dir_set = 2 * I1 * Ne * beta
    pi_set = sim.pi_dir_db(S_dir_set)
    # Create the density grid prior to truncation. 
    # Dimensions: (len(x_set), len(beta))
    if pile is None:
        density_grid = sim.sfs_dir_params(x_set[:, None], 1, S_dir_set) * pi_set
        density_grid += sim.sfs_dir_params(1-x_set[:, None], 1, -S_dir_set) * (1-pi_set)
    else:
        density_grid = sim.sfs_dir_WF_grid(S_dir_set, pile, x_set) * pi_set
        density_grid += sim.sfs_dir_WF_grid(-S_dir_set, pile, 1-x_set) * (1-pi_set)
    # Calculate normalization factors for each effect size
    normalization_factors = calculate_normalization_factors(density_grid, x_set, discovery_x, n_x)
    density_grid = normalize_grid(density_grid, x_set, discovery_x, normalization_factors) 
    return density_grid

# A function to get the conditional density grid for the stabilizing model
def stabilizing_conditional_densities(beta, x_set, discovery_x, I2, Ne, pile=None, n_x=2000):
    S_stab_set = 2 * I2 * Ne * beta**2
    if pile is None:
        density_grid = sim.sfs_ud_params_sigma(x_set[:, None], 1, S_stab_set)
        density_grid += sim.sfs_ud_params_sigma(1-x_set[:, None], 1, S_stab_set)
    else:
        density_grid = sim.sfs_ud_WF_grid(S_stab_set, pile, x_set)
        density_grid += sim.sfs_ud_WF_grid(S_stab_set, pile, 1-x_set)
    # Calculate normalization factors for each effect size
    normalization_factors = calculate_normalization_factors(density_grid, x_set, discovery_x, n_x)
    density_grid = normalize_grid(density_grid, x_set, discovery_x, normalization_factors)
    return density_grid

# A function to get the conditional density grid for the pleiotropic stabilizing model
def pleiotropic_conditional_densities(beta, x_set, discovery_x, Ip, Ne, pile=None, n_x=2000, n_s=1000):
    S_grid = np.logspace(-2, 3, n_s)
    S_p_set = 2 * Ne * Ip * beta**2
    density_grid = np.zeros((len(beta), len(x_set)))
    if pile is None:
        x_densities = sim.sfs_ud_params_sigma(x_set[:, None], 1, S_grid) # (len(x_set), n_s)  
    else:
        x_densities = sim.sfs_ud_WF_grid(S_grid, pile, x_set)
    for ii, S_p in enumerate(S_p_set):
        S_densities = stats.levy.pdf(S_grid, loc=0, scale=S_p) # (n_s,)
        low_prob = stats.levy.cdf(S_grid[0], loc=0, scale=S_p)
        high_prob = 1-stats.levy.cdf(S_grid[-1], loc=0, scale=S_p)
        # Integrate over S densities to get the sfs
        density_grid[ii,:] = np.trapz(x_densities*S_densities, S_grid, axis=1)
        # Add the left and right tails of the S|beta distribution
        density_grid[ii,:] += low_prob * x_densities[:,0]
        density_grid[ii,:] += high_prob * x_densities[:,-1]
    density_grid += np.flip(density_grid, axis=1)
    density_grid = density_grid.T
    # Calculate normalization factors for each effect size
    normalization_factors = calculate_normalization_factors(density_grid, x_set, discovery_x, n_x)
    density_grid = normalize_grid(density_grid, x_set, discovery_x, normalization_factors)
    return density_grid
