import numpy as np
from . import simulation as sim
import scipy.stats as stats
import scipy.special as special

def trad_x_set(min_x, n_points):
    return 1/(1+np.exp(-np.linspace(np.log(min_x/(1-min_x)), np.log((1-min_x)/min_x), n_points)))

log10_sp = np.array([-6, -5.5, -5, -4.5, -4.25, -4, -3.5, -3, -2.5, -2, -1.5])
fp = np.array([0, 26, 117, 230, 250, 233, 136, 56, 22, 13, 13])
simons_ssd_area = np.trapz(x=log10_sp, y=fp)
fp_norm = fp / simons_ssd_area
log10_s_min = np.min(log10_sp)
log10_s_max = np.max(log10_sp)

def simons_ssd(log10_s):
    density = np.interp(log10_s, log10_sp, fp_norm)
    density = np.where((log10_s < log10_s_min) | (log10_s > log10_s_max),
                       np.zeros_like(density), density)
    return density

def simons_ssd_F(log10_s, num=1000):
    log10_s_grids = np.linspace(log10_s_min, log10_s, num)
    return np.trapz(y=simons_ssd(log10_s_grids), x=log10_s_grids, axis=0)

def simons_ssd_F_inv(F, num=1000):
    log10_s_grids = np.linspace(log10_s_min, log10_s_max, num)
    F_grids = simons_ssd_F(log10_s_grids, num)
    return np.interp(F, F_grids, log10_s_grids)

def simons_ssd_sample(nn):
    F = np.random.uniform(size=nn)
    return simons_ssd_F_inv(F)

def logflat_posterior_density(S_p, Ne=10000, S_ud_set=None):
    '''
    Compute the posterior density of scaled selection coefficients S_ud 
    given a scaled selection parameter S_p under the log-flat prior

    Parameters
    ----------
    S_p : float
        Scaled selection parameter under log-flat prior (S_p=2*Ne*I_P*beta^2)
    S_ud_set : ndarray
        Set of S_ud values to compute the density over (S_ud=2*Ne*s_ud)

    Returns
    -------
    density_set : ndarray
        Density of S_p over the set of S_ud values
    mass_lower : float
        Mass of the density below the set of S_ud values
    mass_upper : float
        Mass of the density above the set of S_ud values
    '''
    # Multiply simons s_ud by two to put it on our scale
    s_ud_fullrange = np.logspace(log10_s_min, log10_s_max, 10000)
    # Compute density of S_p over the full range of S_ud 
    # by multiplying the likelihood of S_p times the log-flat prior
    density_fullrange = (stats.gamma.pdf(x=S_p, a=0.5, scale=2*Ne*s_ud_fullrange) / 
                                         (2*Ne*s_ud_fullrange))
    mass_fullrange = np.trapz(y=density_fullrange, x=2*Ne*s_ud_fullrange)
    density_fullrange /= mass_fullrange
    # If given a set of S_ud values, compute the density of S_p over that set
    # as well as the remaining mass on either end
    if S_ud_set is not None:
        # Interpolate the density at the S_ud values
        density_set = np.interp(S_ud_set, 2*Ne*s_ud_fullrange, density_fullrange)
        # Calulate the mass at the lower end
        if S_ud_set[0] > 2*Ne*s_ud_fullrange[0]:
            mass_lower = np.trapz(y=density_fullrange[:np.where(2*Ne*s_ud_fullrange > S_ud_set[0])[0][0]],
                                    x=2*Ne*s_ud_fullrange[:np.where(2*Ne*s_ud_fullrange > S_ud_set[0])[0][0]])
        else:
            mass_lower = 0
        # Calulate the mass at the upper end
        if S_ud_set[-1] < 2*Ne*s_ud_fullrange[-1]:
            mass_upper = np.trapz(y=density_fullrange[np.where(2*Ne*s_ud_fullrange < S_ud_set[-1])[0][-1]:],
                                    x=2*Ne*s_ud_fullrange[np.where(2*Ne*s_ud_fullrange < S_ud_set[-1])[0][-1]:])
        else:
            mass_upper = 0
        return density_set, mass_lower, mass_upper
    else:
        return density_fullrange, 0, 0

def simons_ssd_posterior_density(S_p, Ne, S_ud_set=None):
    '''
    Compute the posterior density of scaled selection coefficients S_ud 
    given a scaled selection parameter S_p under the high-dimensional pleiotropy model

    Parameters
    ----------
    S_p : float
        Scaled selection parameter under high-dimensional pleiotropy model (S_p=2*Ne*I_P*beta^2)
    Ne : int
        Effective population size
    S_ud_set : ndarray
        Set of S_ud values to compute the density over (S_ud=2*Ne*s_ud)

    Returns
    -------
    density_set : ndarray
        Density of S_p over the set of S_ud values
    mass_lower : float
        Mass of the density below the set of S_ud values
    mass_upper : float
        Mass of the density above the set of S_ud values
    '''
    # Multiply simons s_ud by two to put it on our scale
    s_ud_fullrange = np.logspace(log10_s_min, log10_s_max, 10000)
    # Compute density of S_p over the full range of S_ud 
    # by multiplying the likelihood of S_p times the simons ssd prior
    density_fullrange = (stats.gamma.pdf(x=S_p, a=0.5, scale=2*Ne*s_ud_fullrange) * 
                                         simons_ssd(np.log10(s_ud_fullrange)) / # divide by 2 to return to simons scale
                                         (2*Ne*s_ud_fullrange))
    mass_fullrange = np.trapz(y=density_fullrange, x=2*Ne*s_ud_fullrange)
    density_fullrange /= mass_fullrange
    # If given a set of S_ud values, compute the density of S_p over that set
    # as well as the remaining mass on either end
    if S_ud_set is not None:
        # Interpolate the density at the S_ud values
        density_set = np.interp(S_ud_set, 2*Ne*s_ud_fullrange, density_fullrange)
        # Calulate the mass at the lower end
        if S_ud_set[0] > 2*Ne*s_ud_fullrange[0]:
            mass_lower = np.trapz(y=density_fullrange[:np.where(2*Ne*s_ud_fullrange > S_ud_set[0])[0][0]],
                                    x=2*Ne*s_ud_fullrange[:np.where(2*Ne*s_ud_fullrange > S_ud_set[0])[0][0]])
        else:
            mass_lower = 0
        # Calulate the mass at the upper end
        if S_ud_set[-1] < 2*Ne*s_ud_fullrange[-1]:
            mass_upper = np.trapz(y=density_fullrange[np.where(2*Ne*s_ud_fullrange > S_ud_set[-1])[0][0]:],
                                    x=2*Ne*s_ud_fullrange[np.where(2*Ne*s_ud_fullrange > S_ud_set[-1])[0][0]:])
        else:
            mass_upper = 0
        return density_set, mass_lower, mass_upper
    else:
        return density_fullrange, 0, 0

# Function to sample GWAS effect sizes from the Simons et al. (2022) DFE
def simons_gwas_sample(nn, v_cutoff, plei=True, batch_size=10000, Ne=10000, min_x=0.01, n_x=1000, WF_pile=None, gwas_noise=False, pp=5e-8):
    '''
    Simulate a gwas sample using the DFE from Simons et al. (2022)

    Parameters
    ----------
    nn : int
        Number of samples to simulate
    v_cutoff : float
        Cutoff for the GWAS
    plei : bool
        Whether to use the pleiotropy model
    batch_size : int
        Batch size for simulation
    Ne : int
        Effective population size
    min_x : float
        Minimum x to sample
    n_x : int
        Number of points to use in the x grid
    WF_pile : ndarray
        Pileup of WF simulations to use for sampling
    gwas_noise : bool
        Whether to add noise to the GWAS effect sizes
    pp : float
        P-value cutoff, determines the amount of noise to add

    Returns
    -------
    x : ndarray
        Array of derived allele frequencies
    s : ndarray
        Array of selection coefficient values
    b : ndarray
        Array of effect sizes

    '''
    result_x = np.array([], dtype=np.float64)
    result_s = np.array([], dtype=np.float64)
    result_b = np.array([], dtype=np.float64)
    if gwas_noise:
        result_b_hat = np.array([], dtype=np.float64)
    while len(result_x) < nn:
        # Sample x from the equilibrium distribution above min_x
        if WF_pile is None:
            xx, log10_ss = simons_eq_sample_x(batch_size, batch_size, Ne, min_x, n_x)
        else:
            xx, log10_ss = simons_WF_sample_x(batch_size, WF_pile, batch_size, Ne, min_x)
        # sample b
        SS = 10**log10_ss * 2 * Ne
        if plei:
            bb = np.random.normal(loc=0, scale=np.sqrt(SS))
        else:
            bb = np.random.choice([-1, 1], size=len(SS)) * np.sqrt(SS)
        if gwas_noise:
            neff = (2*(special.erfinv(1-pp))**2) / v_cutoff
            print(neff)
            bb_hat = np.random.normal(loc=bb, scale=np.sqrt(1/(2*xx*(1-xx)*neff)), size=len(SS))
            vv_hat = 2*xx*(1-xx)*bb_hat**2
            keep = vv_hat > v_cutoff
        else:
            vv = 2*xx*(1-xx)*bb**2
            keep = vv > v_cutoff
        result_x = np.concatenate((result_x, xx[keep]))
        result_s = np.concatenate((result_s, log10_ss[keep]))
        result_b = np.concatenate((result_b, bb[keep]))
        if gwas_noise:
            result_b_hat = np.concatenate((result_b_hat, bb_hat[keep]))
    if gwas_noise:
        return result_x[:nn], result_s[:nn], result_b[:nn], result_b_hat[:nn]
    else:
        return result_x[:nn], result_s[:nn], result_b[:nn]


def simons_eq_sample_x(nn, batch_size=10000, Ne=10000, min_x=0.01, n_x=1000):
    '''
    Sample x from the equilibrium distribution of the Simons et al. (2022) DFE.
    
    Parameters
    ----------
    nn : int
        Number of samples to return.
    batch_size : int
        Number of samples to draw at a time.
    Ne : float
        Effective population size.
    min_x : float
        Minimum value of x to appear in the sample.
    n_x : int
        Number of points to use in the x grid.

    Returns
    -------
    x : ndarray
        Array of x values.
    s : ndarray
        Array of s values.
    '''
    # Get the x grid
    x_set = trad_x_set(min_x, n_x)
    # Calculate the probability mass at zero
    m0 = np.trapz(y=sim.sfs_ud_params_log(x_set, 1, 0), x=x_set)
    result_x = np.array([], dtype=np.float64)
    result_s = np.array([], dtype=np.float64)
    while result_x.shape[0] < nn:
        # Sample seletion coefficients from ssd
        s_sample = simons_ssd_sample(batch_size)
        SS = 10**s_sample * 4 * Ne # Multiply by 4 here because our sfs is scaled slightly differently
        # Calculate the probability mass at SS
        sfs_grid = sim.sfs_ud_params_log(x_set, 1, SS[:, None])
        sfs_mass = np.trapz(y=np.exp(sfs_grid), x=x_set)
        # Normalize the probability mass to neutral
        SS_probs = sfs_mass / m0
        SS_keep = np.random.uniform(size=batch_size) < SS_probs
        # Remove values we don't use
        SS = SS[SS_keep]
        sfs_grid = np.exp(sfs_grid[SS_keep,:] - np.log(sfs_mass[SS_keep,None]))
        # Compute the CDF
        CDF_grid = np.cumsum((sfs_grid[:,1:]   * np.diff(x_set) + 
                              sfs_grid[:,0:-1] * np.diff(x_set))/2, axis=1)
        CDF_grid = np.pad(CDF_grid, ((0,0),(1,0)), mode="constant", constant_values=0)
        # Sample x using inverse CDF
        x_sample = np.array([np.interp(np.random.uniform(), CDF_grid[ii,:], x_set) for ii in range(SS.shape[0])])
        result_x = np.concatenate((result_x, x_sample))
        result_s = np.concatenate((result_s, s_sample[SS_keep]))
    return result_x[0:nn], result_s[0:nn]

def simons_WF_sample_x(nn, WF_pile, batch_size=10000, Ne=10000, min_x=0.01):
    '''
    Sample x from Wright-Fisher simulations using the Simons et al. (2022) DFE.

    Parameters
    ----------
    nn : int
        Number of samples to return.
    WF_pile : dict
        Dictionary of Wright-Fisher simulations.
    batch_size : int
        Number of samples to draw at a time.
    Ne : float
        Effective population size.
    min_x : float
        Minimum value of x to appear in the sample.

    Returns
    -------
    x : ndarray
        Array of x values.
    s : ndarray
        Array of s values.
    '''
    S_0_ii = np.where(WF_pile["s_set"] == 0)[0][0]
    sfs_grid_WF = WF_pile["sfs_grid"][S_0_ii,:,:]
    interp_x = WF_pile["interp_x"]
    s_ud_wf = np.abs(WF_pile["s_ud_set"])
    s_ud_wf_max = np.max(s_ud_wf)
    # Calculate the probability mass at zero
    m0 = np.trapz(y=sfs_grid_WF[0,:], x=interp_x)
    result_x = np.array([], dtype=np.float64)
    result_s = np.array([], dtype=np.float64)
    while result_x.shape[0] < nn:
        # Sample seletion coefficients from ssd
        s_sample = simons_ssd_sample(batch_size)
        SS = 10**s_sample * 4 * Ne # Multiply by 4 here because our sfs is scaled slightly differently
        # Relate sampled selection coefficients to the grid used in WF calculations
        s_ud_comp = SS/(2*WF_pile["tenn_N"][0])
        s_ud_wf = np.abs(WF_pile["s_ud_set"])
        s_ud_wf_max = np.max(s_ud_wf)
        s_ud_comp[s_ud_comp >= s_ud_wf_max] = s_ud_wf_max - 1/(2*WF_pile["tenn_N"][0]*1000)
        s_ud_ii_upper = np.argmax(s_ud_wf[:,np.newaxis] > s_ud_comp, axis=0)
        s_ud_ii_lower = s_ud_ii_upper - 1

        w_lower = (s_ud_wf[s_ud_ii_upper] - s_ud_comp) / (s_ud_wf[s_ud_ii_upper] - s_ud_wf[s_ud_ii_lower])
        w_upper = 1 - w_lower

        sfs_grid = (w_upper[:,np.newaxis]*sfs_grid_WF[s_ud_ii_upper] +
                    w_lower[:,np.newaxis]*sfs_grid_WF[s_ud_ii_lower])
        sfs_mass = np.trapz(y=sfs_grid, x=interp_x)
        # Normalize the probability mass to neutral
        SS_probs = sfs_mass / m0
        SS_keep = np.random.uniform(size=batch_size) < SS_probs
        # Remove values we don't use
        SS = SS[SS_keep]
        sfs_grid = sfs_grid[SS_keep,:] / sfs_mass[SS_keep, None]
        # Compute the CDF
        CDF_grid = np.cumsum((sfs_grid[:,1:]   * np.diff(interp_x) +
                                sfs_grid[:,0:-1] * np.diff(interp_x))/2, axis=1)
        CDF_grid = np.pad(CDF_grid, ((0,0),(1,0)), mode="constant", constant_values=0)
        # Sample x using inverse CDF
        x_sample = np.array([np.interp(np.random.uniform(), CDF_grid[ii,:], interp_x) for ii in range(SS.shape[0])])
        result_x = np.concatenate((result_x, x_sample))
        result_s = np.concatenate((result_s, s_sample[SS_keep]))
    return result_x[0:nn], result_s[0:nn]
        
