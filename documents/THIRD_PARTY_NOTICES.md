# Third-party notices

The supplied code license is in [LICENSE](../LICENSE). Imported components retain their
source attribution and applicable terms. Backbone weights, trained derivatives and data
carry the terms of their own distributors.

| Work | Use | Attribution and terms |
|---|---|---|
| UNI2-h, Mahmood Lab | Cell backbone with fold-specific adapters; TLS feature source | [Model repository](https://huggingface.co/MahmoodLab/UNI2-h); CC-BY-NC-ND-4.0 as stated there |
| Phikon-v2, Owkin | Cell backbone with adapters; TB image encoder | [Model repository](https://huggingface.co/owkin/phikon-v2) and its model license |
| InstanSeg | Nucleus detection and selected detector initialization | [Source repository](https://github.com/instanseg/instanseg); retain package and selected-model notices |
| PyTorch, torchvision, timm, transformers, scikit-learn, scikit-image, SciPy, NumPy, pandas, pyarrow, OpenCV, tifffile, zarr, h5py, matplotlib, Pillow, imagecodecs, huggingface-hub, OpenSlide | Runtime dependencies | Each package's published license; versions are listed in [requirements.txt](../requirements.txt) and [3d/pipeline/requirements-geometry.txt](../3d/pipeline/requirements-geometry.txt), and OpenCV's in the install commands of [README.md](../README.md) |
| Shapely | 3D nerve area tables (`3d/scripts/nerve_area_table.py`) | BSD-3-Clause; versions in [requirements.txt](../requirements.txt) and [3d/pipeline/requirements-geometry.txt](../3d/pipeline/requirements-geometry.txt) |
| puppeteer-core and a headless Chrome | Optional 3D framework-page smoke test (`3d/pipeline/page_smoke.sh`) | Their published terms; installed separately |
| AnnData and decoupler | TLS program and GC score preparation (`tls/define/scripts/precompute_scores.py`, `precompute_gc_scores.py`) | Their package terms; versions are listed in `tls/define/requirements.txt` |
| SimpleITK and tqdm | Optional B-spline stage of `align/image_registration/scripts/pack_pipeline.py` | Apache-2.0 (SimpleITK) and MPL-2.0/MIT (tqdm); install separately to enable that stage |

Obtain each model from its distributor under that distributor's own terms, and retain the
notices accompanying acquired model and data assets.
