# Add the band-limited score as diagnostic only

Keep the existing full-image `match_score` as the camera-search ranking score.
After the winner is selected, filter its Class Average and Matched Projection
with the same physical Butterworth band-pass and calculate a soft-circular,
weighted zero-mean normalized correlation. Report this Diagnostic Band-Limited
Score with requested and effective frequency, mask, box, pixel-size, and
Nyquist metadata.

The first version does not use this score to select a Class Camera Orientation,
derive Match Confidence, or report a second-best margin. The raw-score beam
search does not exhaustively evaluate the orientation space, so a band-limited
margin over its visited candidates would imply stronger global evidence than
the calculation provides. Frequency and mask defaults remain overridable and
their effective values are recorded rather than treated as universal constants.
