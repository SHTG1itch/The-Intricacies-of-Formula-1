# Random Matrix Theory utilities: Marchenko-Pastur denoising and Wigner spacing test.
#
# Here's the backstory. We were reading about ways to improve covariance matrix estimates
# when you don't have very much data - which is exactly our problem. In the walk-forward
# model we're trying to figure out how "uncertain" we are about each driver's true ability,
# but we only have maybe 5 to 10 races to go on. That's not a lot.
#
# We found a technique that was originally invented for nuclear physics. In the 1950s, Eugene
# Wigner was studying energy levels of uranium nuclei and noticed something strange: the gaps
# between consecutive energy levels weren't random. If you modeled the nucleus as a huge
# random matrix, the eigenvalue spacings followed a very specific distribution:
#     P(s) = (pi/2) * s * exp(-pi * s^2 / 4)
# This is now called the "Wigner surmise." The key thing it tells you is whether a matrix
# looks like pure noise or whether it has real structure in it.
#
# About 10 years later, Marchenko and Pastur (1967) took Wigner's insight further. They worked
# out the math for what the EIGENVALUE DISTRIBUTION looks like for a random covariance matrix.
# If you have a matrix built from T noisy samples of N variables, the eigenvalues of the
# resulting covariance matrix should fall in a specific "bulk" range:
#     [lambda_min, lambda_max] = [sigma^2 * (1-sqrt(q))^2,  sigma^2 * (1+sqrt(q))^2]
# where q = N/T (basically "how few observations do you have relative to how many variables").
# Anything outside this range is REAL signal - it wouldn't show up in a purely noisy matrix.
#
# The practical application is clever: if you eigendecompose your covariance matrix and find
# eigenvalues way above lambda_max, those represent genuine structure in your data. Eigenvalues
# inside the bulk are just estimation noise from having limited data. So you replace the noisy
# eigenvalues with their average (the "constant shrinkage" approach from Ledoit & Wolf 2004)
# and get a cleaner, more accurate matrix.
#
# The connection to our F1 work: we have 20 drivers and maybe 10 races worth of data per season.
# That means N=20 (drivers), T~10 (races) and q = N/T = 2. This is actually very high - it means
# almost everything in our early-season matrices is noise. MP tells us exactly how much to trust.
# Late in the season when T=20, q drops to 1.0, and more eigenvalues start crossing above the
# threshold into "real signal" territory. The denoising automatically gets out of the way as
# more data comes in - we didn't have to tune any schedule for it.
#
# The other use we found: the driver-feature similarity matrix. For each race, we have a
# 20-driver x 54-feature matrix. Applying MP to the 20x20 driver-driver similarity version
# (where q = 20/54 = 0.37) extracts the genuine performance tiers - the eigenvalues that
# represent "Red Bull tier vs Ferrari tier vs midfield" rather than random feature noise.
# This lets us gently regularize predictions toward what the feature structure actually implies.
#
# We tried several approaches that didn't pan out:
# - Applying MP directly to the Plackett-Luce Fisher matrix: the Fisher eigenvalues don't
#   separate into clean "signal + noise" the way MP assumes, so all eigenvalues fell below
#   the threshold and the denoising was too aggressive.
# - Using the full posterior covariance matrix for the dominance z-gate: mathematically
#   correct but the positive off-diagonal entries caused negative var_gap in equal-competition
#   seasons like 2021, which blew up the z-score and messed with the alpha gates.
# The Wigner spacing test on the Fisher matrix runs as a diagnostic only - it correctly
# identifies that our Fisher matrices have structure beyond pure GOE noise, but since
# ALL our seasons show this, it can't discriminate between "VER wins everything" vs "anyone
# can win any race", so we don't use the test result to gate any predictions.
#
# End result: the MP driver-feature clustering is the main thing we use in predictions.
# The Fisher information matrix section uses the full matrix (better math) but the diagonal
# elements only (for stability reasons explained in walk_forward_model.py).
#
# References that were actually useful:
#   Marchenko & Pastur (1967) Math. USSR-Sbornik 1(4):457-483  <- the original paper
#   Wigner (1955) Ann. Math. 62(3):548-564  <- nuclear physics, surprisingly readable
#   Laloux, Cizeau, Bouchaud, Potters (1999) PRL 83:1467  <- first financial application
#   Ledoit & Wolf (2004) JMVA 88(2):365-411  <- where "bulk mean shrinkage" comes from
#   https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution
#   https://en.wikipedia.org/wiki/Wigner_surmise

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import ks_1samp


def marchenko_pastur_edges(q, sigma2=1.0):
    # computes the lower and upper edge of the Marchenko-Pastur bulk distribution
    # q = N/T: the ratio of "how many variables" to "how many observations"
    # sigma2: the noise level (variance of individual entries in the random matrix)
    #
    # anything strictly above lambda_max is real signal - it wouldn't appear in pure noise
    # anything between lambda_min and lambda_max is consistent with random fluctuations
    # when q >= 1 (more variables than observations), lambda_min collapses to zero, which
    # means the matrix is rank-deficient on average - this is a warning sign that you have
    # way more variables than you can reliably estimate from your data
    #
    # ref: Marchenko & Pastur (1967) eq. 4.1
    sq = np.sqrt(float(q))
    lam_max = float(sigma2) * (1.0 + sq) ** 2
    lam_min = float(sigma2) * max(1.0 - sq, 0.0) ** 2   # can't go below zero
    return lam_min, lam_max


def _fit_sigma2_from_bulk(eigenvalues, q, n_grid=150):
    # when we don't know the noise level upfront, this function figures it out by
    # finding the sigma^2 that makes the bulk eigenvalue density look most like MP.
    # it's basically maximum likelihood estimation - which sigma^2 makes the observed
    # bulk eigenvalues most probable under the Marchenko-Pastur distribution?
    #
    # we only look at eigenvalues in the "plausible bulk" region to avoid the signal
    # eigenvalues (the big ones) biasing our noise estimate upward. the idea is that
    # the top few eigenvalues are real signal so we exclude them from the fit.
    # ref: Laloux et al. (1999) used this same approach for stock return covariances
    eigs = np.sort(np.clip(eigenvalues, 0, None))

    def neg_ll(log_s2):
        # we optimize over log(sigma^2) instead of sigma^2 directly because that keeps
        # the search space unconstrained (sigma^2 has to be positive, log(sigma^2) doesn't)
        s2 = np.exp(log_s2)
        lmin, lmax = marchenko_pastur_edges(q, sigma2=s2)
        # restrict to eigenvalues that fall inside the theoretical bulk for this sigma^2
        bulk = eigs[(eigs > lmin + 1e-12) & (eigs < lmax - 1e-12)]
        if len(bulk) < 3:
            return 1e10   # penalize sigma^2 values that leave too few bulk eigenvalues
        # the MP density at each eigenvalue - this is what the density "should" look like
        # if the matrix really were pure noise with this sigma^2
        # formula: f(lambda) = sqrt((lmax-lambda)(lambda-lmin)) / (2*pi*sigma^2*q*lambda)
        # ref: Marchenko & Pastur (1967) eq. 4.3
        inner = np.clip((lmax - bulk) * (bulk - lmin), 0, None)
        denom = 2.0 * np.pi * s2 * q * bulk
        pdf_vals = np.sqrt(inner) / np.clip(denom, 1e-14, None)
        valid = pdf_vals > 1e-12
        if not valid.any():
            return 1e10
        return -float(np.sum(np.log(pdf_vals[valid])))   # negative log-likelihood

    res = minimize_scalar(neg_ll, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(res.x))


def rmt_denoise_psd(M, T_eff, sigma2=None, return_diagnostics=False):
    # cleans up a symmetric positive semi-definite matrix by throwing away the noise eigenvalues
    # and keeping only the genuine signal ones.
    #
    # the idea in plain english: when you build a covariance-like matrix from limited data,
    # some of the "structure" you see is real (like "Red Bull is clearly faster than Williams")
    # and some is just statistical noise from having a small sample. MP tells you where the
    # line is - eigenvalues above lambda_max are real signal, eigenvalues below are noise.
    # for the noisy ones, instead of zeroing them out (which would make the matrix singular
    # and impossible to invert), we replace them with their average - this is the Ledoit-Wolf
    # "oracle approximating shrinkage" idea and it preserves the trace of the matrix.
    #
    # M: the matrix to clean - needs to be symmetric and positive semi-definite
    # T_eff: how many "effective observations" built this matrix. this sets q = n/T_eff.
    #    for the driver-feature similarity matrix we use T_eff = n_features (54)
    #    for the Fisher information matrix we use T_eff = total weighted race positions
    # sigma2: if None, we estimate it from the bulk eigenvalues using _fit_sigma2_from_bulk
    # return_diagnostics: set to True to also get back n_signal and lambda_max
    #
    # ref: the "constant bulk shrinkage" approach comes from Ledoit & Wolf (2004)
    # ref: Bun, Bouchaud & Potters (2017) Physics Reports 666 Section 4.2 has a great
    #      comparison of different denoising approaches if you want to go deeper

    n = M.shape[0]
    M = (M + M.T) / 2.0   # symmetrize first - floating point errors can break this slightly

    # eigh is faster and more numerically stable than eig for symmetric matrices
    # it also guarantees real eigenvalues, which is what we need here
    eigvals, eigvecs = np.linalg.eigh(M)   # returns eigenvalues in ascending order
    eigvals = np.clip(eigvals, 0.0, None)   # force non-negative: PSD matrices shouldn't have negatives

    # q = dimension / effective-observations: this is the key ratio
    # large q means lots of noise, small q means more trustworthy signal
    q = float(n) / max(float(T_eff), 1.0)

    if sigma2 is None:
        # exclude the top 20% of eigenvalues from the sigma^2 estimate - those are probably
        # signal eigenvalues and would inflate the noise estimate if we included them
        n_exclude = max(1, n // 5)
        bulk_eigs = eigvals[:-n_exclude] if n_exclude < n else eigvals
        sigma2 = max(float(np.mean(bulk_eigs[bulk_eigs > 1e-12])), 1e-8)

    _, lam_max = marchenko_pastur_edges(q, sigma2=sigma2)

    # everything above lam_max is real - everything at or below is noise
    is_signal = eigvals > lam_max
    n_signal  = int(is_signal.sum())

    # replace noise eigenvalues with the average of all noise eigenvalues
    # this is better than zero because: (a) zero eigenvalues break matrix inversion,
    # and (b) the mean preserves the total "mass" of the matrix (trace stays the same)
    noise_eigs = eigvals[~is_signal]
    noise_mean = float(np.mean(noise_eigs)) if len(noise_eigs) > 0 else 0.0
    clean_eigvals = np.where(is_signal, eigvals, noise_mean)

    # reconstruct the matrix from cleaned eigenvalues
    # M_clean = V * diag(lambda_clean) * V'
    M_clean = eigvecs @ np.diag(clean_eigvals) @ eigvecs.T
    M_clean = (M_clean + M_clean.T) / 2.0   # symmetrize again - small numerical errors can accumulate

    if return_diagnostics:
        return M_clean, n_signal, lam_max
    return M_clean


def wigner_spacing_test(eigenvalues):
    # tests whether the gaps between eigenvalues follow the "Wigner surmise" - the
    # distribution that Wigner derived for nuclear energy levels in 1955.
    #
    # quick intuition: in a completely random matrix (pure noise), eigenvalues "repel"
    # each other - you almost never see two eigenvalues extremely close together.
    # the exact distribution of the normalized gap s = (spacing / mean spacing) is:
    #     P(s) = (pi/2) * s * exp(-pi * s^2 / 4)
    # notice this goes to zero as s->0 (the "level repulsion" Wigner observed in nuclei)
    # and has a peak around s=1 (most gaps cluster near the average)
    #
    # how we use it: we run this test on the Fisher information matrix from the PL model.
    # if the eigenvalue spacings DON'T follow Wigner's distribution (low p-value), the
    # matrix has real structure beyond pure random noise. the test runs every fit() call
    # and the p-value is stored as a diagnostic, though we found it can't reliably
    # distinguish between "VER wins everything" and "completely random season" because
    # PL Fisher matrices always look structured relative to pure GOE.
    #
    # what the return values mean:
    #   spacings: array of normalized spacing values (s = gap / mean_gap)
    #   ks_stat:  Kolmogorov-Smirnov statistic vs Wigner CDF - smaller = more GOE-like = more random
    #   p_value:  the p-value for the KS test - large p = bulk is consistent with random noise
    #
    # ref: Wigner (1955) Ann. Math. 62:548 - the nuclear physics paper that started all this
    # ref: Porter & Rosenzweig (1960) Suppl. Nuovo Cimento 13:612 - extended validation
    # ref: https://en.wikipedia.org/wiki/Wigner_surmise for an accessible explanation
    eigs = np.sort(np.asarray(eigenvalues, dtype=float))

    raw_spacings = np.diff(eigs)
    raw_spacings = raw_spacings[raw_spacings > 1e-14]   # skip numerically zero gaps

    if len(raw_spacings) < 3:
        # not enough eigenvalues to run a meaningful test - happens at the very start
        # of the season before much data has been accumulated
        return np.array([]), float("nan"), float("nan")

    mean_s = float(np.mean(raw_spacings))
    if mean_s < 1e-14:
        return np.array([]), float("nan"), float("nan")

    # normalize the spacings by the mean - Wigner's result is for the dimensionless ratio
    s = raw_spacings / mean_s

    # the Wigner GOE CDF is F(s) = 1 - exp(-pi/4 * s^2)
    # this is exact for 2x2 GOE matrices and empirically accurate for large N
    # ref: Mehta (2004) "Random Matrices" 3rd ed., Chapter 3 has the full derivation
    def wigner_goe_cdf(x):
        return 1.0 - np.exp(-np.pi / 4.0 * np.asarray(x, dtype=float) ** 2)

    # Kolmogorov-Smirnov test: does our spacing distribution match Wigner's CDF?
    ks_stat, p_value = ks_1samp(s, wigner_goe_cdf)
    return s, float(ks_stat), float(p_value)


def mp_pdf(lam, q, sigma2=1.0):
    # the Marchenko-Pastur probability density at a specific eigenvalue
    # mainly useful for plotting: you can overlay this on a histogram of your
    # empirical eigenvalues to see which ones fall inside the noise bulk and which
    # ones poke above it into "real signal" territory
    # ref: Marchenko & Pastur (1967) eq. 4.3
    lmin, lmax = marchenko_pastur_edges(q, sigma2=sigma2)
    if lam <= lmin or lam >= lmax:
        return 0.0   # outside the support - probability is zero
    numerator   = np.sqrt((lmax - lam) * (lam - lmin))
    denominator = 2.0 * np.pi * float(sigma2) * float(q) * float(lam)
    return float(numerator / max(denominator, 1e-14))
