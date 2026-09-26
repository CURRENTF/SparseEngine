import json
import os
import numpy as np

def load_jsonl(path):
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return None
    data = {}
    with open(path, 'r') as f:
        for line in f:
            try:
                obj = json.loads(line)
                l_idx = obj['layer_idx']

                raw_idx = obj['top_token_idx']


                if isinstance(raw_idx, list) and len(raw_idx) == 1 and isinstance(raw_idx[0], list) and isinstance(raw_idx[0][0], list):
                    heads_indices = [set(h) for h in raw_idx[0]]

                elif isinstance(raw_idx, list) and isinstance(raw_idx[0], list) and len(raw_idx[0]) == 1:
                    heads_indices = [set(h[0]) for h in raw_idx]

                elif isinstance(raw_idx, list) and isinstance(raw_idx[0], list):
                    heads_indices = [set(h) for h in raw_idx]
                else:
                    print(f"Unknown format in {path}: {type(raw_idx)}")
                    continue

                data[l_idx] = heads_indices
            except Exception as e:
                print(f"Error parsing line in {path}: {e}")
    return data

def compare():
    path_my = './my_snapkv.jsonl'
    path_ada = './adakv_snapkv.jsonl'

    data_my = load_jsonl(path_my)
    data_ada = load_jsonl(path_ada)

    if not data_my or not data_ada:
        return

    common_layers = sorted(list(set(data_my.keys()) & set(data_ada.keys())))
    if not common_layers:
        print("No common layers found between the two files.")
        print(f"My layers: {list(data_my.keys())}")
        print(f"Ada layers: {list(data_ada.keys())}")
        return

    print(f"{'Layer':<10} | {'Mean Overlap':<15} | {'Best Offset':<12}")
    print("-" * 45)

    all_overlaps = []

    for l_idx in common_layers:
        heads_my = data_my[l_idx]
        heads_ada = data_ada[l_idx]

        num_heads = min(len(heads_my), len(heads_ada))


        best_offset = 0
        max_avg_overlap = -1


        for offset in range(-128, 129):
            current_layer_overlaps = []
            for h in range(num_heads):
                set_my = heads_my[h]

                set_ada_shifted = {idx + offset for idx in heads_ada[h]}

                intersection = len(set_my & set_ada_shifted)
                union = len(set_my | set_ada_shifted)
                if union == 0:
                    current_layer_overlaps.append(1.0)
                else:

                    current_layer_overlaps.append(intersection / len(set_my))

            avg_overlap = np.mean(current_layer_overlaps)
            if avg_overlap > max_avg_overlap:
                max_avg_overlap = avg_overlap
                best_offset = offset

        all_overlaps.append(max_avg_overlap)
        print(f"{l_idx:<10} | {max_avg_overlap:>15.2%} | {best_offset:<12}")

    print("-" * 45)
    print(f"{'Total Avg':<10} | {np.mean(all_overlaps):>15.2%}")

if __name__ == "__main__":
    compare()
