"""Live acceleration must preserve the explicit finite-grid pair currents."""
import numpy as np
import pytest

from ssrf_realtime.ideal_model import IdealBinModel, IdealBinParams


@pytest.mark.parametrize('bins', [60, 61, 500, 701])
@pytest.mark.parametrize('overlap,cutoff', [('lorentzian', 0.), ('gaussian', 0.),
                                         ('lorentzian', 3.17)])
@pytest.mark.parametrize('cross_ratio', [0., 1., 2.3])
def test_fft_matches_explicit_pair_currents_and_conserves_mass_and_P(bins, overlap, cutoff, cross_ratio):
    model = IdealBinModel(IdealBinParams(
        n_bins=bins, diffusion_overlap=overlap, kernel_cutoff_widths=cutoff,
        cross_branch_ratio=cross_ratio, microwave_diffusion_factor=1.8))
    rng = np.random.default_rng(73)
    model.n = model.mu[:, None]*rng.dirichlet([.5, 2., 3.], bins)
    fast = model._spin_diffusion_terms(dnp_on=True)
    assert model._diffusion_fft is not None
    # The original sparse pair sum is an independent evaluation, with no FFT.
    model._diffusion_fft = None
    direct = model._spin_diffusion_terms(dnp_on=True)
    for name, actual in fast.items():
        np.testing.assert_allclose(actual, direct[name], atol=2e-15, rtol=5e-11)
        np.testing.assert_allclose(actual.sum(axis=1), 0., atol=2e-17)
        assert abs(np.sum(actual[:, 0]-actual[:, 2])) < 2e-15


def test_correlation_cutoff_and_parameter_edits_rebuild_the_correct_operator():
    model = IdealBinModel(IdealBinParams(n_bins=61))
    model._ensure_diffusion_kernel()
    assert model._diffusion_fft is not None
    model.params.orientation_corr_fraction = .3
    model._ensure_diffusion_kernel()
    assert model._diffusion_fft is None
    model.params.orientation_corr_fraction = 0.
    model.params.kernel_cutoff_widths = model.dR/model.params.zq_width_R
    model._ensure_diffusion_kernel()
    assert model._diffusion_fft is None  # Preserve exact floating-point edge decisions.
    model.params.kernel_cutoff_widths = 0.
    model._ensure_diffusion_kernel()
    old = model._diffusion_fft.copy()
    model.params.zq_width_R *= 2
    model._ensure_diffusion_kernel()
    assert not np.array_equal(old, model._diffusion_fft)


def test_live_burn_and_recovery_match_original_pair_evolution():
    params = dict(n_bins=61, rf_response_mode='spectral', rf_peak_selectivity=.7,
                  dnp_enabled=True, t1_rate=.003)
    fast = IdealBinModel(IdealBinParams(**params))
    direct = IdealBinModel(IdealBinParams(**params))
    direct._ensure_diffusion_kernel()
    direct._diffusion_fft = None
    for model in (fast, direct):
        model.start_program()
        model.step(700)  # Includes the program endpoint at t=1.
        model.stop_program()
        model.step(300)
    assert fast.t == direct.t
    np.testing.assert_allclose(fast.n, direct.n, atol=2e-16, rtol=2e-12)
    np.testing.assert_array_equal(fast.delivered_exposure, direct.delivered_exposure)
