"""Persona-header likelihood evaluation (Amram, 2026), applied to the Gemma distress claim.

The method needs no training. It asks a BASE model which of two persona descriptions makes a
piece of text more likely, and reads the answer off the log-likelihood ratio. We use it three
ways, in the order the experiments run:

  1  phle_score.py   + phle_tables.py     score answers under competing headers, AUROC over bags
  2  phle_continue.py                     let the base model continue a bag and read what it wrote
  3  phle_name.py                         leave the description blank and let the base model fill it in

Reference implementation: github.com/OzAmram/psm-mixture-test (scripts/subliminal_score.py,
scripts/make_tables.py). The statistics here are a port of theirs, kept deliberately close:
matched-question bags, cluster bootstrap over questions, within-question label-permutation null,
and a folded band over the wrong headers.
"""
