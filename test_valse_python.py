"""Numerical validation of the paper-equation Python VALSE implementation."""
import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy.special import i0e, i1e

try:
    import valse_python as vp
except ModuleNotFoundError:
    vp = None


class ValseTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(vp, "The Python VALSE implementation must exist")

    def test_inverse_bessel_approximation(self):
        r = np.array([0, .01, .3, .529, .53, .7, .849, .85, .99, .99999])
        k = vp.Ainv(r)
        assert_allclose(i1e(k) / i0e(k), r, atol=.004)

    def test_moments_against_independent_quadrature(self):
        theta = np.linspace(-np.pi, np.pi, 131072, endpoint=False)
        n = np.arange(-8, 9)
        for kappa in [0., .01, 1., 25., 1000.]:
            mu = .731
            density = np.exp(kappa * (np.cos(theta - mu) - 1))
            expected = np.exp(1j * n[:, None] * theta) @ density / density.sum()
            assert_allclose(vp.von_mises_moments(n, mu, kappa), expected,
                            atol=2e-12, rtol=2e-12)

    def test_extreme_moments(self):
        for k in [0., 1e-300, 1e6, 1e12]:
            a = vp.von_mises_moments(np.arange(64), .3, k)
            self.assertTrue(np.isfinite(a).all())
            self.assertTrue(np.max(np.abs(a)) <= 1 + 1e-14)
            self.assertEqual(a[0], 1)
        self.assertTrue(np.isfinite(vp.Ainv(np.array([0, 1]))).all())

    def test_support_updates_against_direct_posterior(self):
        rng = np.random.default_rng(841)
        M, L = 18, 7
        a = rng.normal(size=(M, L)) + 1j * rng.normal(size=(M, L))
        a *= np.sqrt(M / np.sum(np.abs(a)**2, axis=0))
        J = a.conj().T @ a
        signal = a[:, [1, 5]] @ np.array([1.2+.3j, -.8+.7j])
        h = a.conj().T @ signal
        nu, rho, tau = .15, .3, 1.5
        K, s, w, C, trace = vp.maxZ(J, h, M, nu, rho, tau, return_trace=True)
        self.assertEqual(np.flatnonzero(s).tolist(), [1, 5])
        for item in trace:
            ids = item['support']
            precision = J[np.ix_(ids, ids)] / nu + np.eye(len(ids)) / tau
            covariance = np.linalg.inv(precision)
            mean = covariance @ h[ids] / nu
            assert_allclose(item['mean'], mean, atol=2e-12)
            assert_allclose(item['covariance'], covariance, atol=2e-12)
        self.assertTrue(np.all(np.diff([0.] + [t['objective'] for t in trace]) > 0))
        assert_allclose(C[np.ix_(s, s)], C[np.ix_(s, s)].conj().T, atol=1e-14)

    def test_noise_equation_preserves_frequency_uncertainty(self):
        m = np.arange(16)
        a = np.column_stack([vp.von_mises_moments(m, f, k)
                             for f, k in [(.2, 50), (-1.1, 200)]])
        w = np.array([1+.2j, -.7j])
        c = np.array([[.03, .002j], [-.002j, .02]])
        y = a @ w + .1
        j = a.conj().T @ a
        np.fill_diagonal(j, len(m))
        h = a.conj().T @ y
        expanded = (np.vdot(y,y).real - 2*np.vdot(h,w).real
                    + np.vdot(w,j@w).real + np.trace(j@c).real) / len(m)
        residual = np.linalg.norm(y-a@w)**2 / len(m)
        direct = (residual + np.trace(j@c).real / len(m)
                  + np.sum(np.abs(w)**2 * (1 - np.sum(np.abs(a)**2,axis=0)/len(m))))
        assert_allclose(expanded, direct, atol=1e-14)
        assert_allclose(vp.noise_variance_update(y,a,w,c), direct, atol=1e-14)
        self.assertGreater(direct, residual)

    def test_support_deletion_against_direct_posterior(self):
        rng = np.random.default_rng(2)
        M,L = 10,8
        a = rng.normal(size=(M,L))+1j*rng.normal(size=(M,L))
        a[:,2] = a[:,1]+.3*a[:,2]
        a *= np.sqrt(M/np.sum(abs(a)**2,axis=0))
        j = a.conj().T@a
        y = a[:,[1,2,6]]@np.array([1,.5j,.7]) + .1*(rng.normal(size=M)+1j*rng.normal(size=M))
        h = a.conj().T@y
        result = vp.maxZ(j,h,M,.2,.3,1.,return_trace=True)
        self.assertTrue(any(t['action']=='deactivate' for t in result[-1]))
        previous_objective = 0.
        for t in result[-1]:
            ids = t['support']
            covariance = np.linalg.inv(j[np.ix_(ids,ids)]/.2 + np.eye(len(ids)))
            assert_allclose(t['covariance'],covariance,atol=2e-12)
            assert_allclose(t['mean'],covariance@h[ids]/.2,atol=2e-12)
            self.assertGreater(t['objective'],previous_objective)
            previous_objective = t['objective']

    def test_global_covariance_index_against_expected_log_likelihood(self):
        # Noncontiguous support {1,4}; component 4 has local index 1.
        rng = np.random.default_rng(43)
        m = np.arange(10)
        a = np.exp(1j*m[:,None]*np.array([-.9,-.4,.2,.8,1.1]))
        y = rng.normal(size=10)+1j*rng.normal(size=10)
        w = np.zeros(5,complex); w[[1,4]] = [.7+.3j,-.5+.8j]
        c = np.zeros((5,5),complex)
        c[np.ix_([1,4],[1,4])] = [[.2,.03+.04j],[.03-.04j,.3]]
        support = np.array([False,True,False,False,True])
        eta = vp.frequency_update_vector(y,a,w,c,support,4,.4)
        source_eta = vp.frequency_update_vector(y,a,w,c,support,4,.4,covariance_column=1)
        grid = np.linspace(-np.pi,np.pi,101)
        direct, represented = [],[]
        for theta in grid:
            atoms = a[:,[1,4]].copy(); atoms[:,1] = np.exp(1j*m*theta)
            direct.append(-(np.linalg.norm(y-atoms@w[[1,4]])**2
                            +np.trace(atoms.conj().T@atoms@c[np.ix_([1,4],[1,4])]).real)/.4)
            represented.append(np.vdot(eta,np.exp(1j*m*theta)).real)
        assert_allclose(np.array(direct)-direct[0],np.array(represented)-represented[0],atol=1e-12)
        self.assertGreater(np.linalg.norm(eta-source_eta),.1)

    def test_asymptotic_moment_boundary(self):
        from scipy.special import ive
        n = np.arange(64)
        k = 1e6+1
        assert_allclose(vp.bessel_ratio(n,k),ive(n,k)/ive(0,k),atol=1e-11,rtol=1e-11)

    def test_single_harmonic_heuristic(self):
        # Heuristic 2 curvature gives Ainv(exp(-1/(2*strength))).
        a, theta, kappa = vp.heuristic2(np.array([100*np.exp(.73j)]),np.array([1]),full_length=16)
        assert_allclose(theta,.73,atol=1e-14)
        assert_allclose(kappa,vp.Ainv(np.exp(-.005)),atol=1e-12)
        assert_allclose(a,vp.von_mises_moments(np.arange(16),theta,kappa))

    def test_floor_sensitivity_and_source_mode(self):
        rng = np.random.default_rng(919)
        y = np.exp(.39j*np.arange(16))+.02*(rng.normal(size=16)+1j*rng.normal(size=16))
        out = [vp.estimate_valse(y,variance_floor=floor) for floor in [1e-10,1e-12,1e-14]]
        assert_allclose(out[0]['y_estimate'],out[1]['y_estimate'],atol=1e-12)
        assert_allclose(out[2]['y_estimate'],out[1]['y_estimate'],atol=1e-12)
        source = vp.estimate_valse(y,covariance_index='source')
        self.assertTrue(np.isfinite(source['y_estimate']).all())
        self.assertEqual(source['covariance_index'],'source')

    def test_noiseless_single_line(self):
        m = np.arange(16)
        y = (1.2-.4j) * np.exp(1j*m*.371)
        out = vp.estimate_valse(y)
        self.assertTrue(out['converged'])
        self.assertEqual(out['support'].sum(), 1)
        self.assertLess(np.linalg.norm(out['y_estimate']-y)**2/np.linalg.norm(y)**2, 1e-8)
        self.assertLess(abs(np.angle(np.exp(1j*(out['freqs'][0]-.371)))), 1e-5)

    def test_noisy_multiple_lines_and_missing_observations(self):
        rng = np.random.default_rng(121)
        full_m = np.arange(32)
        full_y = np.exp(1j*full_m[:,None]*np.array([-.7, 1.05])) @ np.array([1+.2j,.7-.3j])
        m = np.array([0,1,2,4,5,6,8,9,11,13,14,16,18,20,22,24,26,28,30,31])
        y = full_y[m] + .015*(rng.normal(size=len(m))+1j*rng.normal(size=len(m)))
        out = vp.estimate_valse(y,m)
        self.assertTrue(out['converged'])
        self.assertEqual(out['support'].sum(), 2)
        self.assertLess(np.linalg.norm(out['x_estimate']-full_y)**2/np.linalg.norm(full_y)**2,.002)
        assert_allclose(out['moments'] @ out['amps'], out['x_estimate'])
        assert_allclose(out['x_estimate'][m], out['y_estimate'])

    def test_zero_noise_extreme_scale_and_iteration_cap(self):
        z = vp.estimate_valse(np.zeros(16,complex))
        self.assertEqual(z['support'].sum(),0)
        self.assertTrue(z['converged'])
        rng = np.random.default_rng(8)
        for y in [rng.normal(size=16)+1j*rng.normal(size=16),
                  np.exp(.37j*np.arange(16))*1e-140,
                  np.exp(.37j*np.arange(16))*1e140]:
            out = vp.estimate_valse(y, max_iter=200)
            self.assertTrue(np.isfinite(out['x_estimate']).all())
            self.assertTrue(np.isfinite(out['gain_covariance']).all())
            self.assertTrue(np.isfinite(out['noise_var']))
            self.assertGreater(out['noise_var'],0)
        capped = vp.estimate_valse(np.exp(.37j*np.arange(16)),max_iter=1)
        self.assertEqual(capped['iterations'],1)
        self.assertFalse(capped['converged'])

    def test_invalid_input(self):
        for y,m in [(np.ones(4),[0,1,1,2]),(np.ones(4),[-1,0,1,2]),
                    (np.array([1,np.nan]),None),(np.ones(3),[0,.5,1])]:
            with self.assertRaises(ValueError):
                vp.estimate_valse(y,m)


if __name__ == '__main__':
    unittest.main(verbosity=2)
