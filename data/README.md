# FCCT / IRI data

All experiment assets are stored in `data/fcct/`:

- `images/`: 1,000 original COCO val2014 images.
- `trace.jsonl`: 1,371 positive object queries over 500 tracing images.
- `pope.jsonl`: 9,000 original COCO POPE questions over 500 disjoint images.
- `sources/`: the paper, original POPE questions, and COCO instance annotations.
- `dataset_card.json`: selection protocol, counts, licenses, and manifest checksums.
- `sources.lock.json` and `model.lock.json`: source and checkpoint checksums.

This tracing subset was independently selected with seed 17. It is not the
unavailable original author subset. Images, dimensions, sample IDs, checksums,
and tracing/evaluation image separation are checked by `python fcct.py check`.

To fetch missing assets and reconstruct the manifests:

```powershell
.\.venv\Scripts\python.exe scripts/setup_fcct.py --paper --pope --annotations-fast --model
.\.venv\Scripts\python.exe fcct.py prepare --download-images
```

Image acquisition uses the official COCO bucket. Images retain their original
Flickr licenses; consult the original COCO metadata and
[COCO terms](https://cocodataset.org/#termsofuse). Original POPE files are pinned
to the upstream revision recorded in the source lock. Do not commit downloaded
images, annotations, model weights, or experiment outputs to source control.
