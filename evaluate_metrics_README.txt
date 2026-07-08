evaluate_metrics.py — how it works and what it expects
========================================================

PURPOSE
-------
Standalone offline evaluation script. It compares a folder of your generated
(enhanced) images against the SID Sony ground-truth long-exposure images and
reports the three metrics used in the paper:

    - PSNR  (Peak Signal-to-Noise Ratio)     — via scikit-image
    - SSIM  (Structural Similarity)          — via scikit-image
    - LPIPS (Learned Perceptual Image Patch
             Similarity, AlexNet backbone)   — via the `lpips` package

This is separate from inference.py's own --eval flag (which computes the
same metrics in-memory during generation, on un-quantized float predictions).
Use evaluate_metrics.py when you already have generated images saved to
disk as ordinary PNG/JPG files — e.g. from a manual run, a checkpoint you're
comparing after the fact, or a completely different model — and just want to
score a folder of outputs against ground truth without re-running inference.

Note: because this script reads generated images back from disk, it scores
whatever precision they were saved at (usually 8-bit PNG). inference.py's
built-in --eval computes metrics on the un-quantized float output before it's
written to disk, so its numbers will be very slightly better (less
quantization noise). This is expected and normal for a "final" offline check.


WHERE GROUND TRUTH COMES FROM
------------------------------
By default, ground truth is read from config.yaml's data_root/long_dirname,
i.e. ./Sony/long — the raw .ARW long-exposure files. The script demosaics
these on the fly (same camera-WB, no-auto-bright, 16-bit settings used
everywhere else in this project) — it does NOT require the short_rgb /
long_rgb / short_packed_cache folders that training.py builds, so it works
even if you've deleted those caches (as we just did).

You can also point --gt_dir at a folder of already-demosaiced images
(PNG/JPG/etc.) instead of raw .ARW files, if you'd rather not re-demosaic
each time.


NAMING SCHEME THIS SCRIPT EXPECTS
-----------------------------------
There are two matching modes, controlled by --match_mode:

1) "scene" (the default)
   ------------------------
   Every file (both ground truth and generated) is expected to contain the
   5-digit SID scene id somewhere in its filename — the script grabs the
   first run of 4-6 digits it finds. Ground truth and generated files whose
   scene ids match are paired up.

   This is deliberately compatible with inference.py's own output naming
   (test_<short-exposure-stem>.png) without any renaming:

       ground truth : Sony/long/00001_00_10s.ARW           -> scene id 00001
       generated    : results/test_00001_00_0.04s.png       -> scene id 00001
       generated    : results/test_00001_01_0.1s.png        -> scene id 00001
                                                              (both match the
                                                               same GT file)

   This many-to-one matching is intentional: the SID dataset has several
   short-exposure bursts sharing one long-exposure ground truth, so it's
   normal and expected for one GT file to be scored against several
   generated images.

2) "stem" (--match_mode stem)
   -----------------------------
   Strict one-to-one matching: a generated file is only paired with a
   ground-truth file if their basenames (filename without extension) are
   IDENTICAL. Use this if you generate one output per scene and save it
   with the exact same stem as its ground-truth file, e.g.:

       ground truth : Sony/long/00001_00_10s.ARW
       generated    : results/00001_00_10s.png

   This is the "generated image stored in a similar format [to the ground
   truth]" convention — same name, just a normal image file instead of raw
   sensor data.

Accepted extensions:
   - Ground truth : .ARW, .CR2, .NEF, .DNG, .RAF  (raw, demosaiced on the fly)
                    or .png/.jpg/.jpeg/.bmp/.tif/.tiff/.webp (used as-is)
   - Generated     : .png/.jpg/.jpeg/.bmp/.tif/.tiff/.webp  (never raw)

Any ground-truth file with no matching generated image (or vice versa) is
skipped with a warning printed at the end — it will NOT crash the run.

If a generated image's resolution doesn't match its ground truth, it's
resized (bicubic) to the ground truth's resolution before scoring, with a
warning printed — this should only happen if you cropped/resized outputs
yourself; normal full-frame inference output already matches GT resolution.

For LOE only, there's a third folder involved (see LOE's own section below):
--input_dir, the short-exposure inputs, default config.yaml's data_root/
short_dirname (./Sony/short). A generated file is matched to its input by
stripping a known output prefix (test_ or compare_, from inference.py's own
naming) and looking for an exact stem match in --input_dir; if that fails it
falls back to scene-id matching (ambiguous if a scene has several bursts —
a warning is printed and the first candidate is used).


USAGE
-----
Basic (scene-id matching, GT from config.yaml default):

    python evaluate_metrics.py --gen_dir ./results

Exact-stem matching, explicit GT folder:

    python evaluate_metrics.py --gen_dir ./results --gt_dir ./Sony/long --match_mode stem

Add extra metrics (see "other metrics" below):

    python evaluate_metrics.py --gen_dir ./results --extra_metrics mae,dists,loe,fid

Output:
   - Per-pair metrics printed to the console as they're computed.
   - A summary (mean +/- std across all matched pairs) at the end.
   - A CSV report written to <gen_dir>/metrics_report.csv (override with
     --output_csv), with one row per matched pair, plus a trailing AVERAGE row.
   - FID (if requested) is printed separately at the end, NOT added to the
     CSV — see why in its own section below.


OTHER METRICS
-------------
The paper's three (PSNR, SSIM, LPIPS) are implemented directly. Four more
are wired in as optional (--extra_metrics), off by default:

  - MAE   : mean absolute pixel error in [0,1] space. Cheap, interpretable,
            same quantity the L1 loss term already optimizes for. Per-pair,
            appears as its own CSV column.

  - DISTS : Deep Image Structure and Texture Similarity — a learned
            full-reference metric like LPIPS but explicitly modeling texture
            statistics as well as structure. Directly relevant given this
            repo's own DISTS-loss experiment. Needs `pip install DISTS-pytorch`
            (not installed by default) — the script detects it automatically
            if present and warns (without crashing) if you request it and
            it's missing. Per-pair, appears as its own CSV column.

  - LOE (Lightness Order Error) : from the LIME paper (Guo et al., 2017).
            Unlike the other metrics, LOE does NOT compare against ground
            truth — it compares the generated image against the ORIGINAL
            short-exposure INPUT, measuring whether their relative pixel
            lightness order is preserved. A low LOE means the enhancement
            didn't introduce unnatural local contrast reversals (e.g.
            brightening a shadow past a highlight that should stay above
            it). Both images are downsampled to 50x50 (--loe_size) before
            the pairwise comparison, since it's O(N^2) in pixel count — this
            is the same downsampling the original paper uses. Per-pair,
            appears as its own CSV column (blank for any pair where no
            matching input file could be found — see --input_dir above).
            No extra dependency needed — just numpy/opencv, already installed.

  - FID (Frechet Inception Distance) : DIFFERENT KIND OF METRIC from the
            other four — it's not computed per image pair, it's computed
            once over the whole matched SET of ground-truth images vs. the
            whole matched set of generated images, by comparing the mean and
            covariance of their Inception-v3 feature distributions. This is
            why it isn't a CSV column: there is only one FID value per run,
            printed separately at the end. It needs a reasonably large
            sample to be meaningful — the literature typically uses 2000+
            images; with the SID Sony val/test split sizes (dozens to a few
            hundred scenes) the script will still compute a number, but
            prints a warning that it should be read as indicative only, not
            a paper-comparable score. Needs `pip install pytorch-fid` (this
            has been installed into the `llie` conda env already).

