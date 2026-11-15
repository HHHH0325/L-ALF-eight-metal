# Frozen GBR model

Obtain the pretrained model from the paper's code/data link or from its
authors. The source URL and release commit are unverified. Use the original
artifact without retraining or substitution.

Place the authorized file here as `gradient_boosting_model.joblib` and verify
its SHA-256 before running:

```powershell
Get-FileHash .\data\oracle\gradient_boosting_model.joblib -Algorithm SHA256
```

Expected SHA-256: `82e679b0007423696105148fa3a61d23ead541df5cc3bf9535d24daf7f462a82`.
Source: Zhang et al., *A multimodal robotic platform for multi-element
electrocatalyst discovery*, Nature 647, 390-396 (2025),
[DOI: 10.1038/s41586-025-09640-5](https://doi.org/10.1038/s41586-025-09640-5).
