"""Synthetic cluster light curves built from isochrone populations + empirical TESS variability.

See README.md.  The package has two halves:

* population + variability model (``config``, ``empirical``, ``populations``, ``variability``)
  -- a re-implementation of the ``Amaya_Dropbox/tess_clusters/synthetic_populations`` v2
  construction with the issues listed in the README fixed;
* TESS sampling (``tess_sampling``) -- pushes the noise-free cluster signal through the
  existing ``sytheticdata_generate_compare.synth.generate`` machinery (real timestamps,
  exposure averaging, real per-point errors, optional calibrated red noise).
"""
