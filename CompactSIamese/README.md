# Compact Siamese Network 

Reimplementation and retrieval extension of Fojcik et al.'s extremely compact
Siamese video representation, trained and validated on VCDB, then evaluated for the
near-duplicate video retrieval and detection task on FIVR-200K subsets.

##  Overall flow

### 1. Data preparation (VCDB)
1. `data/precompute_fragments.py` — preprocesses VCDB videos into 15-frame,
   32x32 fragments based on interframe difference curve maxima and  writes
   `fragment_metadata.csv`. Core of the method proposed by Fojcik et al.. 

2. `data/precompute_vcdb_segment_pairs.py` — reads VCDB `annotation/*.txt` +
   `fragment_metadata.csv` to build `positive_pairs.csv`.

3. `data/compute_negatives.py` — reads `positive_pairs.csv`, mines hard
   negatives (same video, non-duplicated segment) and easy negatives
   (random other video), writes `negative_pairs.csv`.
   
4. `data/dataset_segment_pairs.py` (`VCDBSegmentPairDataset`) — wraps the
   resulting pair CSVs as a PyTorch `Dataset` for training/evaluation.

### 2. Training
- `train.py` — trains the siamese network as defined in `model/pytorch_model.py` on the segment pair dataset defined in 
  `VCDBSegmentPairDataset` and  builds video-disjoint train/val/test splits. Metrics are logged  via `model/metrics.py`.
- `tune.py` — Optuna wrapper around `train.py` for tuning. 

### 3. VCDB classification evaluation (Experiment 1)
- `test.py --mode segment` — evaluates the trained model on fragment pairs and writes accuracy/precision/recall/F1/F-beta to `metrics.json`. (`test.py` also supports `whole_video_fragments`, `whole_video_descriptors`, and `retrieval` modes, but `segment` is the one used for this experiment.)


### 4. Retrieval extension (FIVR)
1. `data/precompute_fivr_descriptors.py` extracts descriptors for FIVR
   fragments via the trained network. 

2. **`FIVR_evaluation.py`** is the main evaluation script. It supports
   three ways of scoring how similar two videos are:
   - **pred_max** — uses the trained prediction head, taking the single
     best-matching fragment pair between two videos (the default, and what
     the thesis's main results table reports)
   - **pred_chamfer** — same prediction head, but averages over the best
     matches with chamfer similarity across all fragments instead of just the single best one
   - **chamfer** — a simpler baseline that compares raw frame descriptors
     directly, with chamfer similarity, without the trained prediction head

   If desired:`retrieval/pool_embeddings.py` and
   `retrieval/chamfer_retr_faiss.py` offer a  FAISS-based search, we didnt include it in the thesis and did not find it to be particularly good. 


3. `tune_threshold_fivr.py` finds the detection threshold that maximizes
   F-beta on held-out validation queries. 

4. `eval_siam_detect.py` is the **detection** counterpart to
   `FIVR_evaluation.py`. Where `FIVR_evaluation.py` computes retrieval
   metrics, `eval_siam_detect.py` applies a similarity threshold for
   detection metrics. 



### Shared infrastructure
- `runtime_config.py` — path resolution and `get_device()`, used by nearly
  every script above.
