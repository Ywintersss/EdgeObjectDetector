# SDD Progress — RPC Domain-Gap Fix + Coral

Plan: docs/superpowers/plans/2026-07-05-rpc-domain-gap-and-coral.md
Branch: feature/synthetic-scenes
Branch base: b43ae9a

## Tasks
- [x] Task 1: pytest scaffolding (git init + branch done in bootstrap at b43ae9a)
- [x] Task 2: synth_utils geometry
- [x] Task 3: ClassBalancedSampler
- [x] Task 4: compositing utils + yaml writer
- [x] Task 5: extract_cutouts.py (GrabCut)
- [x] Task 6: harvest_backgrounds.py
- [x] Task 7: compose_scenes.py (parallelized)
- [ ] Task 8: full generation + retrain + validate (HUMAN CHECKPOINT — long runs)
- [ ] Task 9: export_edgetpu.py (GATED on Task 8)

## Log
(bootstrap) repo init, .gitignore excludes data trees + weights, feature branch created @ b43ae9a
Task 1: complete (commits b43ae9a..20e41c7, review clean)
Task 2: complete (commits 20e41c7..b2d2971, review clean)
Task 3: complete (commits b2d2971..f6a4713, review clean)
Task 4: complete (commits f6a4713..aac92f3, approved w/ Minors)
Task 5: complete (commits aac92f3..3ec2511, approved; trial made 30 real cut-outs). Minor: _read_single_box only reads line[0] (fine — train labels verified 1 box each).
Task 6: complete (commits 3ec2511..f549cc2, incl. fix f549cc2). Important test-gap fixed (hardened _patch_free direct test). Minor left: edge-adjacent boundary only tested on left side (non-blocking); unreadable-image skip is silent.
Task 7: complete (commits f549cc2..8702129, incl. fix 8702129). Core scene composition. Fixed Critical (--workers/parallelized via sequential pre-sampling → class balance preserved) + Important (worker try/except batch-safety; yaml name derives from --out; dataset_synth*.yml gitignored) + Minors. Re-review clean, no regressions.

Task 8 DATA-GEN: complete. 8000 cut-outs across 200/200 classes; 4000 backgrounds; 25373 train (20000 synth + 5373 single-item); real val(6000)/test(24000) copied. dataset_synth.yml valid. Box-overlay spot check PASSED. NOTE: extract_cutouts took ~1h51m (GrabCut on full-res 1751px imgs — slower than estimated); total data-gen ~2h. Exit code -1 was a process-tree artifact; log shows full success.

## STOPPED FOR THE DAY (resume tomorrow) — 2026-07-06
Task 8 TRAINING was running, STOPPED by user at epoch 11 of rpc_yolo11n_synth. Shells stopped, GPU freed, no orphan processes.
- Data is fully generated & on disk (dataset_synth/, dataset_synth.yml) — skip straight to training tomorrow.
- yolo11n_synth run: reached epoch 11. best.pt = epoch 6 (mAP50 0.154, val_cls ~4.08). last.pt = epoch 11.
- yolo11s probe: NEVER STARTED (was chained after the n-run).
- EARLY READ (epochs 1-11): synthetic data helped ~2x (mAP50 ~0.12-0.15 vs old ~0.065; precision ~0.32 vs 0.15; val_cls ~4.2 flat, no longer diverging to 6). BUT plateauing + train/val cls gap widening = still overfitting to synthetic domain; only PART of the gap closed. Likely limited by synth realism (checkerboard bg, cut-out edges, scale/lighting).
- RESUME OPTIONS tomorrow: (a) let a fresh full run finish to see true early-stop ceiling; (b) run yolo11s probe to test if ceiling is capacity vs data; (c) improve synth realism (smooth backgrounds, denser/occluded scenes) and regenerate. train.py has no --resume flag; either re-run fresh or `yolo train resume model=runs/detect/rpc_yolo11n_synth/weights/last.pt`.

## Task 9 (Coral export) still GATED on a satisfactory Task 8 accuracy result.
## Known cosmetic note: synth backgrounds are visibly checkerboard-tiled (128px patches, light blur). Products segment cleanly & labels correct; fine for testing the hypothesis. Candidate future refinement if realism needs boosting.

## Minor findings (for final-review triage)
- Task 4: no test for NEGATIVE x/y overhang in alpha_paste (impl verified correct by trace; brief called out the case). Worth a 1-test add — relevant since random_placement produces negative coords used in Task 7.
- Task 4: random_placement floor-division can put center up to 0.5px outside canvas (negligible; verbatim from brief).
- Pre-existing em-dash in synth_utils.py module docstring line 1 (predates task 4; conflicts w/ ASCII-hyphen convention).

## ============================================================
## NEW PLAN: Coarse-Category MVP (200->17)
## Plan: docs/superpowers/plans/2026-07-06-coarse-category-mvp.md
## Spec: docs/superpowers/specs/2026-07-06-coarse-category-mvp-design.md
## Branch base for this plan: 8702129
## ============================================================
- [x] Task 1: complete (commits 8702129..f03ef4e, review clean)
- [x] Task 2: complete (commits f03ef4e..52c9587, review clean)
- [x] Task 3: complete (commits 52c9587..<see task-3-report.md>). Generated dataset_synth_coarse/
  (200->17 classes), proved original 200-way labels byte-unchanged (MD5 identical before/after),
  spot-checked coords, dry-ran `check_det_dataset` (NO training). Found + fixed a real bug during
  Step 6: `link_images` symlinked/junctioned the whole `images/` dir, so Ultralytics'
  `Path.resolve()` collapsed the coarse image path back to the ORIGINAL `dataset_synth/images/...`,
  which made the `/images/`->`/labels/` swap silently resolve to the ORIGINAL 200-way labels
  instead of the coarse 17-way ones (no error -- just wrong labels). Fixed by changing
  `link_images` to per-file hardlinks under real (non-symlinked) split directories, so no path
  component under `dataset_synth_coarse/images/` is a reparse point. Re-ran full pipeline after
  the fix; all checks green. Training NOT started (per task scope) -- see staged commands below.
- [ ] Task 4+: not yet planned. Ready-to-run (staged, NOT executed) training commands:
  - `python train.py --data dataset_synth_coarse.yml --name rpc_coarse17_n`
  - `python train.py --model yolo11s.pt --data dataset_synth_coarse.yml --name rpc_coarse17_s`
  - Full-granularity fallback (unchanged): `python train.py --data dataset_synth.yml`
Task 2 Minor findings (for final-review triage):
- write_coarse_yaml header comment hardcodes "17 coarse categories" regardless of len(coarse_names) (inherited from plan; correct for our actual 17-cat use).
- _verify has dead ternary: label = names[c] if isinstance(names,dict) else names[c] (both branches identical; harmless).
Task 3 Minor findings (for final triage):
- link_images OSError fallback catches any OSError (not just cross-volume) and the fallback path is unexercised (same-volume run); acceptable as a loud failure.
- link_images idempotency: `if dst_images.exists(): return` early-out won't pick up new source files on rerun (pre-existing semantic, not a regression).
COARSE PLAN COMPLETE: dataset_synth_coarse/ generated (nc=17), reversibility proven (orig labels byte-unchanged), check_det_dataset resolves to coarse labels. Training NOT started (per instruction). Staged commands recorded above.

## FINAL WHOLE-BRANCH REVIEW (opus): READY TO MERGE — no Critical/Important.
Guarantees verified incl. Ultralytics path-resolution checked against installed source. All prior Minors: DEFER.
New Minors (deferred, consistent w/ existing dataset_synth.yml pattern):
- yaml emits `test: images/test` but main() only remaps train+val; harmless now (no test labels exist; mirrors dataset_synth.yml). Latent if a test split is ever added.
- --verify checks nc/index/histogram but not image<->label path resolution; could fold check_det_dataset behind --verify as an enhancement.

## POST-RUN DIAGNOSIS (rpc_coarse17_n, full run, early-stopped ep35, best ep15)
- Coarse-17 nano: best mAP50 0.231 / mAP50-95 0.136 / P 0.406 / R 0.269 @ imgsz512. Only ~25% over
  200-class nano (0.185); 200-class 11s probe (0.277) still beats it. val/cls_loss still DIVERGES
  (2.98->3.94) -> coarsening removed class-confusion but not the ceiling.
- Confusion matrix (normalized): dominant failure is the BACKGROUND ROW -- 0.49-0.78 of true objects
  per class predicted as background (missed). Recall-limited, NOT confusion-limited. Small/thin items
  (gum 0.78, milk 0.74, candy 0.70 missed) worst hit.
- Val probe on SAME best.pt, imgsz sweep (no retrain): recall 0.269(512) -> 0.306(640) -> 0.323(768);
  mAP50 0.231 -> 0.280 -> 0.298. CONFIRMS small-object scale is a real root cause. Diminishing returns
  past 640 (512->640 +0.037 R; 640->768 +0.017 R) -> 640 is the edge sweet spot. Residual ceiling
  remains (R still ~0.32 @768) = domain-realism problem -> compose_scenes.py, separate work.

## STAGED FOR TOMORROW (decided, NOT run): retrain coarse nano at imgsz=640
- `python train.py --data dataset_synth_coarse.yml --imgsz 640 --batch 16 --name rpc_coarse17_n_640`
- batch 16 chosen (not default 32): 640 raises activation mem ~1.56x -> batch 32 would OOM on 8GB 5060.
- No code change to train.py needed: --imgsz/--batch flags already exist and thread through. Expect
  ~0.32-0.36 mAP50 with meaningfully higher recall vs today's 0.231.

## REAL-SCENE DOMAIN ADAPTATION (plan 2026-07-08) — execution
Task 1 (scene split): complete (commit e728554, review clean). Minors: unused imports in build_real_coarse.py + test header — these are the module's forward imports/test header the plan front-loads; consumed by Tasks 2-5. No fix needed.
Task 2 (guard preprocessing): complete (commit 5fa98f4, review clean). Minor: preprocessing.py EOF newline (pre-existing, cosmetic).
Task 3 (materialize_split): complete (commit 72feca8, review clean). Minors (all verbatim-from-brief): symlink-fallback has no own error handling; .jpg extension hardcoded; symlink-fallback branch untested.
Task 4 (yaml + verify_no_leak): complete (commits 3e12a82 + fix 28587b3, re-review clean). Fixed Important: added test_write_single_item_yaml_structure (was untested — plan's test block omitted it). Minor left: single-item yaml sets train=val placeholder (intentional; YOLO needs a train key).
Task 5 (CLI main + gitignore): complete (commits ff46b84 + fix 09be88f, re-review clean). Fixed Important: idempotency now completion-sentinel-based (.complete written only after full build) so a crash-interrupted run can't be silently treated as done + staging guard reachable; dropped unused --studio-yaml flag. Wiring verified: blend yaml uses COARSE trees only, paths trace correctly.

## REAL-SCENE DOMAIN ADAPTATION — trees generated (NO training run)
- dataset_real/ (real_ft ~18k / real_eval ~3k / reserve ~3k), studio_coarse/ (train ~10k / eval ~2k).
- Reversibility proven: dataset_synth/labels/train MD5 identical before/after (hash: 361a0a34170d56c02736ff4557f80c81).
- check_det_dataset('dataset_real_blend.yml') -> nc 17, resolves to coarse label trees.

STAGED (decided, NOT run) — B0 blended baseline + lever ladder:
- B0: python train.py --data dataset_real_blend.yml --imgsz 640 --batch 16 --name rpc_real_blend_b0
- L1 (occlusion aug):  add copy_paste=0.3 mixup=0.1 (mosaic on) once train.py exposes them / via cfg
- L2 (cls gain):       raise cls loss gain 0.5 -> ~1.0
- L3 (imgsz 768):      python train.py --data dataset_real_blend.yml --imgsz 768 --batch 8 --name rpc_real_blend_l3
- L4 (capacity):       python train.py --model yolo11s.pt --data dataset_real_blend.yml --imgsz 640 --batch 8 --name rpc_real_blend_l4
- Single-item eval per run: YOLO val on eval_single_item.yml with the run's best.pt.
- Original-val (secondary) eval per run: YOLO val on dataset_synth_coarse.yml (its val is the coarse-remapped real 6k val; disjoint from test-derived real_ft).
