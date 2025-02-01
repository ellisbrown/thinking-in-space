#!/usr/bin/env python3

import os
import glob
import json
import math
import csv
import argparse
import datetime


def safe_mean(values):
    """Compute the mean of a list, ignoring any None/nan entries."""
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return float('nan')
    return sum(vals) / len(vals)

def parse_model_args(model_args_str):
    """
    Parse strings like:
      'pretrained=checkpoint_name,conv_template=qwen_1_5,...,max_frames_num=32'
    Return a dict with:
      {
        'pretrained': 'checkpoint_name',
        'max_frames_num': '32',
         ...
      }
    """
    out = {}
    # Split by commas
    parts = model_args_str.split(',')
    for part in parts:
        # Each part is something like 'pretrained=...', 'max_frames_num=32', etc.
        if '=' in part:
            key, val = part.split('=', 1)
            out[key.strip()] = val.strip()
    return out

def main(glob_pattern, output_csv):
    """
    Collect vsibench results from logs into a single CSV.

    :param glob_pattern: str, pattern used to find result JSONs (e.g. 'logs/*/vsibench/*/results.json')
    :param output_csv: str, path to output CSV
    """

    # Prepare CSV header
    header = [
        "timestamp",
        "datestr",
        "model",
        "ckpt",
        "max_frames",
        "git_hash",
        "Overall",
        "Appearance Order",
        "Measurement Avg",
        "Abs Dist",
        "Obj Size",
        "Room Size",
        "Layout Avg",
        "Obj Count",
        "Rel Dist",
        "Rel Dir",
        "Route Plan"
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # Search for all results.json using the user-provided glob pattern
        all_results = glob.glob(glob_pattern)

        # sort by date
        all_results.sort(key=lambda x: os.path.getmtime(x))  # sorts using the last modified timestamp

        for result_file in all_results:
            # Example path: logs/20250106/vsibench/0107_0118_internvl2_8b_8f_internvl2_model_args_746c50/results.json
            path_parts = result_file.split(os.sep)
            # path_parts = ['logs', '20250106', 'vsibench', '0107_0118_internvl2_8b_8f_internvl2_model_args_746c50', 'results.json']
            if len(path_parts) < 2:
                # In case the path doesn't match the expected structure
                continue

             # Get the full timestamp (last modified time) in human-readable ISO format.
            file_timestamp = datetime.datetime.fromtimestamp(os.path.getmtime(result_file)).isoformat()

            datestr = path_parts[1]

            # Load JSON
            with open(result_file, "r") as rf:
                data = json.load(rf)

            # Safely extract top-level git_hash
            git_hash = data.get("git_hash", "N/A")

            # Safely extract config items
            config = data.get("config", {})
            model = config.get("model", "N/A")
            model_args_str = config.get("model_args", "")
            parsed_args = parse_model_args(model_args_str)

            ckpt = parsed_args.get("pretrained", "N/A")
            max_frames = parsed_args.get("max_frames_num", "N/A")

            # Now parse the vsibench scores
            vsibench_data = data.get("results", {}).get("vsibench", {})
            vsibench_score = vsibench_data.get("vsibench_score,none", float('nan'))

            # Initialize sub-scores (default None -> will treat as nan if old format)
            overall = None
            appearance_order = None
            abs_dist = None
            obj_size = None
            room_size = None
            obj_count = None
            rel_dist = None
            rel_dir = None
            route_plan = None

            if isinstance(vsibench_score, dict):
                # new format
                overall = vsibench_score.get("overall", float('nan'))
                appearance_order = vsibench_score.get("obj_appearance_order_accuracy", float('nan'))
                abs_dist = vsibench_score.get("object_abs_distance_MRA:.5:.95:.05", float('nan'))
                obj_size = vsibench_score.get("object_size_estimation_MRA:.5:.95:.05", float('nan'))
                room_size = vsibench_score.get("room_size_estimation_MRA:.5:.95:.05", float('nan'))
                obj_count = vsibench_score.get("object_counting_MRA:.5:.95:.05", float('nan'))
                rel_dist = vsibench_score.get("object_rel_distance_accuracy", float('nan'))
                rel_dir = vsibench_score.get("object_rel_direction_accuracy", float('nan'))
                route_plan = vsibench_score.get("route_planning_accuracy", float('nan'))
            elif isinstance(vsibench_score, float):
                # old format
                overall = vsibench_score
            else:
                # unexpected format
                overall = float('nan')

            # Compute derived metrics
            # Measurement Avg = average(Abs Dist, Obj Size, Room Size)
            measurement_avg = safe_mean([abs_dist, obj_size, room_size])

            # Layout Avg = average(Obj Count, Rel Dist, Rel Dir, Route Plan)
            layout_avg = safe_mean([obj_count, rel_dist, rel_dir, route_plan])

            # Convert Nones to nan for CSV
            def none_to_nan(x):
                return float('nan') if x is None else x

            row = [
                file_timestamp,
                datestr,
                model,
                ckpt,
                max_frames,
                git_hash,
                none_to_nan(overall),
                none_to_nan(appearance_order),
                none_to_nan(measurement_avg),
                none_to_nan(abs_dist),
                none_to_nan(obj_size),
                none_to_nan(room_size),
                none_to_nan(layout_avg),
                none_to_nan(obj_count),
                none_to_nan(rel_dist),
                none_to_nan(rel_dir),
                none_to_nan(route_plan),
            ]

            writer.writerow(row)

    print(f"Done. CSV written to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect vsibench results from logs into a single CSV."
    )
    parser.add_argument(
        "--glob_pattern",
        default="logs/*/vsibench/*/results.json",
        help="Glob pattern to search for results.json files (default: 'logs/*/vsibench/*/results.json')."
    )
    parser.add_argument(
        "--output_csv",
        default="logs/vsibench_results.csv",
        help="Name of the output CSV file (default: 'vsibench_results.csv')."
    )

    args = parser.parse_args()
    main(
        glob_pattern=args.glob_pattern,
        output_csv=args.output_csv
    )
