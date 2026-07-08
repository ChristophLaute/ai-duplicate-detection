import sys, json, time, argparse
import h5py
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "HAP-VR"))


from datasets.generators import HDF5DatasetGenerator
from model.similarity_network import SimilarityNetwork
from evaluation import calculate_similarities_to_queries


THRESHOLDS = {
    "all": -0.2084,
    "nd":   0.3290,
}



def load_id_list(path):
    with open(path) as f:
        return [Path(line.strip()).stem for line in f if line.strip()]
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str, required=True)
    parser.add_argument("--hdf5",     type=str, required=True)
    parser.add_argument("--queries",  type=str, default=None)
    parser.add_argument("--refs",     type=str, default=None)
    parser.add_argument("--output",   type=str, default="results.json")
    parser.add_argument("--mode",     type=str, default="all", choices=["all", "nd"])

    args = parser.parse_args()
    args.features_only = False
    
    threshold = THRESHOLDS[args.mode]

    m = torch.load(args.model, map_location="cpu", weights_only=False)
    sim_network = SimilarityNetwork[m["args"].similarity_network].get_model(**vars(m["args"]))
    sim_network.load_state_dict(m["model"])
    sim_network = sim_network.cuda().eval()

    if args.queries:
        query_ids = load_id_list(args.queries)
    else:
        with h5py.File(args.hdf5, 'r') as f:
            query_ids = list(f.keys())
    ref_ids = load_id_list(args.refs) if args.refs else query_ids
    query_generator = HDF5DatasetGenerator(args.hdf5, query_ids)

    query_loader = DataLoader(query_generator, batch_size=1, num_workers=4)
    queries, query_ids_loaded = [], []
    offline_time = 0.0
    online_time  = 0.0

    for (video_tensor,),(video_id,) in tqdm(query_loader, desc="Loading queries"):
        if video_id and video_tensor.shape[0] > 0:
            t = time.perf_counter()
            feat = sim_network.index_video(video_tensor.cuda())
            online_time += time.perf_counter() - t
            queries.append(feat)
            query_ids_loaded.append(video_id)

    ref_gen = HDF5DatasetGenerator(args.hdf5, ref_ids)
    ref_loader = DataLoader(ref_gen, batch_size=1, num_workers=4)
    similarities = {q:{} for q in query_ids_loaded}

    for (video_tensor,),(video_id,) in tqdm(ref_loader, desc="Processing references"):
        if video_id and video_tensor.shape[0] > 0:
            t_off = time.perf_counter()
            feat = sim_network.index_video(video_tensor.cuda())
            offline_time += time.perf_counter() - t_off

            t_on = time.perf_counter()
            sims = calculate_similarities_to_queries(sim_network, queries, feat, args)
            online_time += time.perf_counter() - t_on

            for i, qid in enumerate(query_ids_loaded):
                similarities[qid][video_id] = float(sims[i])

    print(f"Offline time (ref indexing): {offline_time:.2f}s")
    print(f"Online time (query indexing + comparator): {online_time:.2f}s | per query: {online_time/max(len(query_ids_loaded),1):.3f}s")
    predictions = {}
    for qid, refs in similarities.items():
            matches = sorted(
                [{"ref": r, "score": s} for r, s in refs.items() if r != qid and s >= threshold],
                key=lambda x: x["score"], reverse=True,
            )
            if matches:
                predictions[qid] = matches

    with open(args.output, "w") as f:
            json.dump(predictions, f, indent=2)
    print(f"Saved predictions for {len(predictions)} queries to {args.output}")

if __name__ == "__main__":
    main()
