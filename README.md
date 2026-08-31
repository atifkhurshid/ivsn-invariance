# Invariance properties of predicted eye movements in visual search models

## Abstract
We assess how geometric and photometric transformations affect predicted gaze patterns using computational models that generate fixation sequences via target-driven priority maps. We evaluated the IVSN model (Zhang et al., Nat. Comm., 2018) and a variant with the initial stages replaced by a Gabor filter bank ($IVSN_{Gabor}$). Object arrays display 6 or 8 objects arranged in circular or grid-like layouts, with transformations applied to the target object in the search image. The mean number of fixations to locate the target across 300 trials measures the performance. IVSN achieved stronger baseline performance (6-object array; 2.4 ± 1.6 vs. 2.7 ± 1.7), though $IVSN_{Gabor}$ was less impaired by transformations overall. Blur, noise, and skew did not affect either model but scale and rotation significantly degraded performance in both models, with IVSN performance at 180o rotation being close to random guessing (3.2 ± 1.7). Scale produced an asymmetric effect, with larger targets easier to locate than smaller ones (2.2 ± 1.5 vs 2.8 ± 1.6). Analysis of internal representations showed that the effect of transformations was smaller in higher layers of the models, suggesting that invariance develops over the course of processing in a hierarchical representation. These results show that the models are not equally invariant to all transformations. Distinct computational mechanisms may be necessary to achieve invariance across transformation types.

## Installation
TODO: To be added.

## Experiments
TODO: To be added.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{khurshid2026invariance,
  title   = Invariance properties of predicted eye movements in visual search models},
  author  = {Khurshid, Atif and Kohn, Matthias and Neumann, Heiko},
  booktitle = {European Conference on Eye Movements},
  year    = {2026},
  note    = {A. Khurshid and M. Kohn contributed equally}
}
```

Atif Khurshid*, Matthias Kohn*, Heiko Neumann. "Invariance properties of predicted eye movements in visual search models." ECEM. 2026.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
