# Third-party notices

The supplied code license is in [LICENSE](../LICENSE). Imported components retain their
source attribution and applicable terms. Backbone weights, trained derivatives and data
carry the terms of their own distributors.

| Work | Use | Attribution and terms |
|---|---|---|
| UNI2-h, Mahmood Lab | Cell backbone with fold-specific adapters; TLS feature source | [Model repository](https://huggingface.co/MahmoodLab/UNI2-h); the inherited notice identifies CC-BY-NC-ND-4.0 |
| Phikon-v2, Owkin | Cell backbone with adapters; TB image encoder | [Model repository](https://huggingface.co/owkin/phikon-v2) and its model license |
| InstanSeg | Nucleus detection and selected detector initialization | [Source repository](https://github.com/instanseg/instanseg); retain package and selected-model notices |
| PyTorch, torchvision, timm, transformers, scikit-learn, scikit-image, SciPy, NumPy, pandas, pyarrow, OpenCV, tifffile, zarr, h5py, matplotlib, Pillow, imagecodecs, huggingface-hub, OpenSlide | Runtime dependencies | Each package's published license; versions are listed in [requirements.txt](../requirements.txt) |
| AnnData and decoupler | Preparation of the acquired TLS molecular-score sidecars | Their package terms apply when regenerating those upstream scores |
| SimpleITK and tqdm | Optional B-spline stage of the standalone alignment pack | Apache-2.0 (SimpleITK) and MPL-2.0/MIT (tqdm); install separately to enable that stage |

Obtain each model from its distributor under that distributor's own terms, and retain the
notices accompanying acquired model and data assets.
