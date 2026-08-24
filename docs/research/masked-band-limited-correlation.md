# Masked band-limited correlation for class-camera diagnostics

Research target: a diagnostic score for the CryoSPARC 5.0.6 class-to-camera
matching job. The current implementation uses a full-image, mean-subtracted
normalized correlation to rank candidate cameras. This note evaluates adding a
band-limited, softly masked score without changing that ranking in v1.

Research date: 2026-08-24.

## Decision summary

1. **Do not treat 80--15 Å as a universal scientific default.** It is a
   reasonable broad starting band for some maps, but the useful band depends
   on matching-grid pixel size, Nyquist limit, box extent, map/class quality,
   and particle shape. Make both limits configurable and validate them against
   the actual matching grid. The default may remain 80 Å (low-resolution edge)
   to 15 Å (high-resolution edge), but it must be clipped or rejected with an
   explicit warning when the grid cannot represent it.
2. **Use the same physical Fourier filter for the class average and matched
   projection, then compute a weighted NCC inside a soft circular window.**
   The window is a score weight, not a binary image to multiply before taking
   the global mean. Remove the weighted DC component after filtering.
3. **Do not canonize “45% radius + 5% cosine edge” as universal.** First-party
   sources support a soft/cosine edge, but not one universal radius. Define
   whether radius is relative to full box width or half-width, prefer
   physical/pixel units in metadata, and expose the values for tuning.
4. **Keep the existing match_score and raw-search ranking unchanged.** Add a
   band_limited_score as diagnostic evidence. It should not silently change
   the selected camera or be labelled a probability.
5. **Do not report a band-limited global confidence margin in v1.** A margin
   computed only over candidates visited by the raw-score beam search is a
   margin within an evaluated subset, not evidence that all other orientations
   lose. If a margin is added later, name its candidate-set scope and evaluate
   that set consistently.

These are engineering recommendations derived from the source facts below; the
sources do not prescribe this project's exact score or defaults.

## What the sources establish

### Physical frequency limits are grid-dependent

CryoSPARC describes the Nyquist limit as twice the image pixel size and notes
that features smaller than that are aliased. Its Fourier-space explanation
also gives frequency-pixel spacing as 1 / (N * pixel_size), where N is the box
size in pixels. See the official
[CryoSPARC aliasing guide](https://guide.cryosparc.com/cryo-em-foundations/image-formation/aliasing).

The current project already carries a physical pixel_size on MatchingGrid.
Therefore, interpret the band in cycles per Å, not fixed FFT-index radii:

~~~text
R_low  = 80 Å   -> f_low  = 1 / 80  cycles/Å  (remove longer wavelengths)
R_high = 15 Å   -> f_high = 1 / 15  cycles/Å  (remove shorter wavelengths)
passband: f_low <= |f| <= f_high
Nyquist resolution: R_nyquist = 2 * pixel_size
box extent: L = N * pixel_size
Fourier-bin spacing: Δf = 1 / L
~~~

The band is representable only if R_low > R_high, the high-resolution edge
does not demand frequencies beyond Nyquist, and the discrete grid contains a
non-empty set of usable frequency bins. A requested low-resolution edge can be
longer than the box extent; then the grid has no Fourier samples at those very
long wavelengths. Report that as box-limited rather than silently presenting
an exact physical band.

CryoSPARC examples make the same point operationally: Fourier cropping changes
pixel size and therefore the Nyquist limit, and the same nominal resolution
setting has different pixel implications at different box sizes
([Fourier-cropping case study](https://guide.cryosparc.com/processing-data/tutorials-and-case-studies/case-study-end-to-end-processing-of-an-inactive-gpcr-empiar-10668),
[Downsample Particles](https://guide.cryosparc.com/processing-data/all-job-types-in-cryosparc/extraction/job-downsample-particles)).

### Band-pass correlation is established practice, but tuning is data-specific

The Bsoft first-party protocol defines a band-pass as selecting a frequency
range between two resolution limits. It describes cross-correlation with
resolution limits as a way to avoid noise and emphasize frequencies with useful
signal-to-noise ratio. It also warns that filtering can create
oscillations/fringes, particularly when the high-resolution limit is pushed
too far ([Heymann, Bsoft: Image Processing for Structural Biology](https://doi.org/10.21769/BioProtoc.4393),
[full protocol PDF](https://bio-protocol.org/pdf/Bio-protocol4393.pdf)).

The Bsoft example uses 5--50 Å limits for one alignment example, not 80--15 Å
as a general rule. In a recent primary template-matching study, performance
was measured while varying high-pass, low-pass, mask size, voxel size, and
angular sampling. The authors conclude that optimal values depend on data
quality, object size, and object shape and should be tuned systematically
([Turo et al., Nature Communications 2024](https://www.nature.com/articles/s41467-024-47839-8),
especially Fig. 2 and the parameter-assessment text).

Thus 80--15 Å is defensible only as a configurable initial diagnostic band for
the present data. It is not defensible as a fixed assumption for other
CryoSPARC projects, resolutions, or particle sizes.

### Smooth filters and soft masks reduce, but do not eliminate, artifacts

CryoSPARC's Volume Tools documentation states that a rectangular low-pass
filter can introduce ringing, while a Butterworth filter reduces ringing at
the cost of a less abrupt cutoff. It also states that lower filter order
reduces ringing but produces a slower frequency falloff
([CryoSPARC Volume Tools](https://guide.cryosparc.com/processing-data/all-job-types-in-cryosparc/utilities/job-volume-tools)).

CryoSPARC's mask guide explains that a hard spatial edge becomes oscillatory
in Fourier space and can cause alignment to follow the mask artifact. It
recommends a soft edge and explicitly says ideal softness is dataset- and
subvolume-dependent
([CryoSPARC mask creation](https://guide.cryosparc.com/processing-data/tutorials-and-case-studies/mask-selection-and-generation-in-ucsf-chimera)).
The CryoSPARC membrane-protein guide likewise says masks used in 2D-to-3D
processing should be smooth, without sudden cliffs.

RELION documents the same cosine soft-edge mechanism, with soft-edge widths
specified in pixels and tuned by inspecting the result. Its tutorial uses
15 Å as a useful low-pass value for making smooth solvent masks for many
proteins, but does not make that value a universal alignment band or give a
universal mask-radius fraction
([RELION mask creation and post-processing](https://relion.readthedocs.io/en/latest/SPA_tutorial/Mask.html)).

These sources support this distinction:

- **Filter:** a frequency-domain weighting applied identically to both
  compared images. A smooth Butterworth or raised-cosine transition is safer
  than a rectangular brick-wall band.
- **Score mask:** a spatial weight defining which pixels contribute to the
  diagnostic statistic. A cosine-tapered edge is appropriate; a hard binary
  edge is not.

The sources do not establish a single mandatory order for this diagnostic. The
least surprising v1 order is:

~~~text
class average, matched projection
        -> same Fourier bandpass
        -> inverse FFT
        -> weighted masked NCC (soft circular weights)
~~~

This makes frequency content comparable before the spatial region is scored.
The mask should not fill the outside with zeros and then use a global-mean NCC;
those zeros become part of the statistic. If boundary leakage is observed, add
a separately tested pre-window/padding step rather than adding it implicitly.

### A soft edge has support in pixels, not just a percentage

CryoSPARC mask parameters and RELION examples express soft padding in pixels.
CryoSPARC additionally gives a resolution-dependent rule of thumb for some FSC
masks, while noting that ideal softness must be determined empirically. Those
recommendations are for 3D FSC/refinement masks, not a direct prescription for
this 2D diagnostic.

CryoSPARC extraction guidance commonly starts with an extraction box about
twice the particle diameter
([start-to-finish guide](https://guide.cryosparc.com/live/new-live-session-start-to-finish-guide)).
A radius occupying 45% of the full box width would have an outer diameter of
about 90% of the box, leaving little edge margin. It may be useful for a large,
centered object, but it is not a safe universal assumption and may admit
substantial solvent/background. “45% radius” is also ambiguous unless the
implementation says whether the denominator is full box width, half-width, or
particle diameter.

Engineering implication: expose both inner radius and taper width, and record
them in pixels and Å. A percentage can be a convenience default, but its
meaning must be explicit. A parameter sweep around a structure-derived radius
(enough to contain the class signal plus margin) is more defensible than
freezing 45% for all maps.

## Recommended v1 definition

### Inputs and validation

Use the MatchingGrid arrays and common physical pixel size. Let N be the square
2D grid size and a its pixel size in Å/pixel. Accept:

~~~text
low_resolution_A   # larger wavelength edge, e.g. 80
high_resolution_A  # smaller wavelength edge, e.g. 15
mask_radius        # explicit units: pixels or Å; fraction only as a wrapper
mask_edge_width    # explicit units: pixels or Å
filter_type        # smooth Butterworth or raised-cosine by default
filter_order       # recorded, not hidden
~~~

Validate and report:

1. low_resolution_A > high_resolution_A > 0.
2. high_resolution_A >= 2*a for strict Nyquist validity. If the requested edge
   is finer than Nyquist, either clamp it to Nyquist with a warning or mark the
   band invalid; do not claim that missing frequencies were compared.
3. L = N*a, Δf = 1/L, and the number of radial Fourier bins in the actual
   passband. A band with no usable bins is invalid.
4. The mask has nonzero support and the taper does not extend beyond the image.
   If it does, clip only with an explicit metadata warning.

### Frequency response

Construct the radial frequency grid with physical spacing. SciPy documents that
fftfreq(n, d=pixel_size) returns frequency-bin centers in cycles per unit of
the sample spacing
([SciPy fftfreq](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fftfreq.html)).

For a smooth band, use a product of a high-pass and low-pass response. One
possible Butterworth form is:

~~~text
f = sqrt(fx**2 + fy**2)
f_low  = 1 / low_resolution_A
f_high = 1 / high_resolution_A

H_highpass(f) = 1 / (1 + (f_low / max(f, eps))**(2*order))
H_lowpass(f)  = 1 / (1 + (f / f_high)**(2*order))
H(f) = H_highpass(f) * H_lowpass(f)
~~~

The exact response is an implementation choice. Important invariants are that
both inputs use the same response, DC is attenuated by the low-frequency edge,
and the response is not a brick wall unless tests show ringing is acceptable.
SciPy documents the Butterworth -3 dB cutoff and numerical considerations for
its IIR form
([SciPy butter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html)).
This project can build the response directly on its FFT grid; an IIR
implementation is not required.

### Soft circular weighted NCC

Build a centered radial weight w once. Let r_inner be full-weight radius and
r_outer = r_inner + edge_width the radius where the weight reaches zero. Use a
cosine taper:

~~~text
w(r) = 1                              if r <= r_inner
       0.5 * (1 + cos(pi*t))          if r_inner < r < r_outer
       0                              if r >= r_outer
where t = (r - r_inner) / edge_width
~~~

After filtering class x and projection y, calculate moments using the weights:

~~~text
W    = sum(w)
mu_x = sum(w * x) / W
mu_y = sum(w * y) / W

xc = x - mu_x
yc = y - mu_y
num = sum(w * xc * yc)
den = sqrt(sum(w * xc**2) * sum(w * yc**2))
score = num / den
~~~

Clip only tiny floating-point overshoots to [-1, 1]. Do not calculate means
over the full box after zeroing outside the mask. That computes a different
statistic whose value depends on the amount of masked background. This weighted
formulation is consistent with the masked-registration literature: simply
applying a mask can change the correlation statistic unless the mask's influence
is accounted for
([Padfield, Masked object registration in the Fourier domain](https://doi.org/10.1109/TIP.2011.2181402),
and the [scikit-image masked-registration explanation](https://scikit-image.org/docs/stable/auto_examples/registration/plot_masked_register_translation.html)).

Even when the high-pass response suppresses DC, weighted mean removal should
remain. It protects against residual DC, filter-transition leakage,
interpolation offsets, and nonzero edge/background values.

Return an invalid diagnostic (null plus a reason) when any of these hold:

- W is zero or below a documented numerical floor;
- there are no usable passband bins;
- either weighted variance is below a documented floor;
- the numerator or denominator is non-finite.

Do not turn an undefined score into 0.0 or 1.0; those values imply information
that the metric did not have.

## Relationship to the existing search

The current solver obtains a translation and raw full-image match_score, then
uses that raw score to rank evaluated beam candidates. For diagnostic-only v1:

1. Preserve existing translation, camera, match_score, beam expansion, and
   selected winner.
2. Apply the bandpass and weighted mask only after the raw winner and its
   already-computed matched projection are available.
3. Store band_limited_score and all effective filter/mask metadata beside the
   existing camera metadata.

This answers whether the chosen match remains similar when the score focuses on
a specified physical band and central region. It does not answer which camera
the band-limited metric would have selected. That should be a separate opt-in
search mode with its own tests.

### Why a second-best band margin is risky now

The current beam search keeps only a small number of raw-score winners at
intermediate expansion stages. Even if every visited candidate is retained in
a local dictionary, the visited set is not an exhaustive orientation grid. A
band-limited second-best score over that set can be a local diagnostic, but it
is not evidence that all other orientations lose. It may also be incomparable
with the raw margin when the metrics prefer different candidates.

For v1, either omit band_limited_score_margin or report it only with an explicit
name such as band_limited_margin_within_raw_evaluated_candidates, plus:

~~~text
candidate_set = raw_search_evaluated
candidate_count
raw_winner_band_score
band_winner_candidate_id (if computed)
~~~

Do not feed that margin into match_confidence until separate validation
establishes calibration against known synthetic and experimental cases.

## Metadata to record

For every class result with a diagnostic score, record enough information to
reproduce and audit it:

~~~json
{
  "band_limited_score": 0.0,
  "band_limited_score_valid": true,
  "band_limited_invalid_reason": null,
  "band_low_resolution_A_requested": 80.0,
  "band_high_resolution_A_requested": 15.0,
  "band_low_resolution_A_effective": 80.0,
  "band_high_resolution_A_effective": 15.0,
  "band_filter_type": "butterworth",
  "band_filter_order": 4,
  "band_passband_bin_count": 0,
  "matching_pixel_size_A": 0.0,
  "matching_box_size": 0,
  "matching_box_size_A": 0.0,
  "matching_nyquist_resolution_A": 0.0,
  "mask_shape": "soft_circle",
  "mask_radius_px": 0.0,
  "mask_edge_width_px": 0.0,
  "mask_weight_sum": 0.0,
  "mask_effective_pixel_count": 0.0,
  "score_definition": "weighted_zero_mean_ncc_after_fourier_bandpass",
  "score_role": "diagnostic_only",
  "candidate_set_scope": "raw_search_winner"
}
~~~

Exact field names can follow the repository's JSON convention. Important points
are to distinguish requested from effective limits, preserve the common physical
grid, state the score definition, and make the diagnostic role explicit.

## Validation plan (TDD)

Implement tests before code changes. The smallest useful test set is:

1. **Identity:** identical class and projection give approximately 1.0 for both
   raw and band-limited scores.
2. **Affine intensity:** adding a constant and multiplying by a positive scalar
   does not change weighted zero-mean NCC.
3. **Band selectivity:** a known low-frequency-only mismatch changes raw score
   but is strongly attenuated when the low-frequency edge excludes it; a
   high-frequency noise component above the high-resolution edge is likewise
   attenuated.
4. **Physical scaling:** grids with different pixel sizes but the same physical
   sinusoidal component use the same Å cutoffs and retain/reject it consistently.
5. **Nyquist guard:** a requested high-resolution edge finer than 2*a is clamped
   or invalid according to the documented policy, with metadata.
6. **Box/bin guard:** an empty or nearly empty discrete passband returns an
   invalid diagnostic rather than a fabricated finite score.
7. **Soft-edge behavior:** the taper is monotonic from 1 to 0; changing edge
   width changes only weighting, not raw score or camera winner.
8. **Mask normalization:** compare the weighted formula with a hand-computed
   reference; ensure zero-filled outside pixels do not change the result beyond
   floating-point tolerance.
9. **Diagnostic isolation:** construct candidates for which raw and band scores
   disagree; verify selected camera and existing match_score remain raw-score
   results in v1.
10. **Real-data audit:** run the J1025/J1083 case, export score distributions
    and metadata, and inspect whether the band score is less uniformly high
    without using a fixed threshold as a correctness claim.

The real-data audit should compare several bands (for example 100--20, 80--15,
and a band clipped to matching-grid Nyquist) and several mask widths. The
purpose is sensitivity analysis, not selecting whichever setting produces the
prettiest score.

## Sources

- [CryoSPARC Guide: Aliasing](https://guide.cryosparc.com/cryo-em-foundations/image-formation/aliasing)
- [CryoSPARC Guide: Volume Tools](https://guide.cryosparc.com/processing-data/all-job-types-in-cryosparc/utilities/job-volume-tools)
- [CryoSPARC Guide: Mask creation](https://guide.cryosparc.com/processing-data/tutorials-and-case-studies/mask-selection-and-generation-in-ucsf-chimera)
- [CryoSPARC Guide: Membrane-protein mask guidance](https://guide.cryosparc.com/processing-data/tutorials-and-case-studies/tutorial-tips-for-membrane-proteins)
- [CryoSPARC Guide: Start-to-finish extraction guidance](https://guide.cryosparc.com/live/new-live-session-start-to-finish-guide)
- [RELION: Mask creation and post-processing](https://relion.readthedocs.io/en/latest/SPA_tutorial/Mask.html)
- [Heymann, Bsoft: Image Processing for Structural Biology](https://doi.org/10.21769/BioProtoc.4393)
- [Bsoft protocol PDF](https://bio-protocol.org/pdf/Bio-protocol4393.pdf)
- [Turo et al., High-confidence 3D template matching for cryo-electron tomography](https://www.nature.com/articles/s41467-024-47839-8)
- [SciPy fftfreq](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fftfreq.html)
- [SciPy butter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html)
- [Padfield, Masked object registration in the Fourier domain](https://doi.org/10.1109/TIP.2011.2181402)
- [scikit-image masked normalized cross-correlation example](https://scikit-image.org/docs/stable/auto_examples/registration/plot_masked_register_translation.html)

