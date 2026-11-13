"""Independent checks of the CP control's physical and statistical assumptions."""
import numpy as np
from numpy.testing import assert_allclose
from cp_confirmation import cp_noise_statistic, cp_signal_audit
from pilot_ofdm import cn


def test_clean_cp_cancels_signal_for_all_profiles():
    assert max(cp_signal_audit().values()) < 2e-12


def test_cp_variance_and_units():
    rng=np.random.default_rng(338)
    # 12,000 independent groups: four differences per block, T=4.
    v=.37
    nf=np.sqrt(v)*cn(rng,(12000,4,64))
    cp=np.sqrt(v)*cn(rng,(12000,4,4))
    vh=cp_noise_statistic(nf,cp,blocks=4)
    assert abs(vh.mean()/v-1)<.01
    assert abs(vh.var()/(v*v/16)-1)<.04
    # This identity also verifies the shared useful-tail noise convention.
    manual=np.mean(abs(cp-np.fft.ifft(nf,axis=-1,norm='ortho')[...,-4:])**2/2,axis=(1,2))
    assert_allclose(vh,manual,atol=0,rtol=0)
