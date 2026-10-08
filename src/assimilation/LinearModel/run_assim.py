"""ES-MDA on a linear model, compared with the exact posterior.

The state is a 1D field of 150 cells with a Gaussian prior, and the data are noisy
observations of every fifth cell. The model is linear and everything is Gaussian, so
the exact posterior is known (the Kalman filter solution), and the ensemble posterior
from ES-MDA can be checked against it. The simulator is lin_1d from PET, which just
reads the field at the observed cells, so the case runs in seconds.

Run write_data.py first to (re)create the observations.
"""

import numpy as np
import matplotlib.pyplot as plt
from geostat.decomp import Cholesky
from input_output import read_config
from pipt import ESMDA
from simulator.simple_models import lin_1d


def main():
    kwda, kwsim, kwens = read_config.read('config.toml')

    esmda = ESMDA(kwda, kwens, lin_1d(kwsim))
    res = esmda.run_assimilation()
    print(f'\nData misfit: {res.prior_data_misfit:.1f} -> {res.data_misfit:.1f}\n')

    # ---------------------------------------------------------------
    # Exact posterior, to check ES-MDA against
    #
    # The model is linear and the prior and data errors are Gaussian, so the
    # posterior is Gaussian too, and its mean and covariance follow from the
    # Kalman formulas. No ensemble is involved: this is the answer ES-MDA
    # should approach as the ensemble grows.
    # ---------------------------------------------------------------
    prior = esmda.ensemble.prior_info['permx']
    nx = prior['nx']

    # Prior covariance, as the prior ensemble was drawn from
    Cx = Cholesky().gen_cov2d(nx, prior['ny'], prior['variance'][0], prior['corr_length'][0],
                              prior['aniso'][0], prior['angle'][0], prior['vario'][0])

    # The model observes the field at the report points
    G = np.zeros((len(kwsim['reportpoint']), nx))
    G[np.arange(G.shape[0]), kwsim['reportpoint']] = 1.0

    d_obs = esmda.ensemble.obs_vector
    Cd = np.diag(esmda.ensemble.obs_variance)

    # Exact posterior mean and standard deviation (Kalman gain K)
    K = Cx @ G.T @ np.linalg.inv(G @ Cx @ G.T + Cd)
    x_prior = np.asarray(prior['mean'], dtype=float)
    x_post_exact = x_prior + K @ (d_obs - G @ x_prior)
    std_post_exact = np.sqrt(np.diag(Cx - K @ G @ Cx))

    # ---------------------------------------------------------------
    # Compare the ES-MDA posterior ensemble with the exact posterior
    # ---------------------------------------------------------------
    cells = np.arange(nx)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    ax1.plot(cells, x_prior, color='grey', label='prior')
    ax1.plot(cells, res.x.mean(axis=1), label='ES-MDA posterior')
    ax1.plot(cells, x_post_exact, color='C3', linestyle='--', label='exact posterior')
    ax1.plot(kwsim['reportpoint'], d_obs, 'k.', label='data')
    ax1.set_ylabel('mean')

    ax2.plot(cells, np.sqrt(prior['variance'][0]) * np.ones(nx), color='grey', label='prior')
    ax2.plot(cells, res.x.std(axis=1, ddof=1), label='ES-MDA posterior')
    ax2.plot(cells, std_post_exact, color='C3', linestyle='--', label='exact posterior')
    ax2.set_ylabel('standard deviation')
    ax2.set_xlabel('cell')

    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
    ax1.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.25))

    fig.tight_layout()
    fig.savefig('results.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
