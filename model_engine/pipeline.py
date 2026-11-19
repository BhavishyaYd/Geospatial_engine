
from pathlib import Path
import argparse
import json
import time

import torch
from PIL import Image

from csdnet.inference import (
    load_model,
    predict_change,
    save_results
)

from csdnet.extract_regions import extract_regions

from remoteclip.temporal_comparison import (
    RemoteCLIPTemporalComparison
)

from remoteclip.semantic_analysis import (
    RemoteCLIPSemanticAnalysis
)

from florence2.analyze_csdnet_regions import analyze_regions


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "pipeline_runs" / "latest"
)


def release_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_pipeline(
    image_t1_path,
    image_t2_path,
    output_dir,
    min_area=30,
    padding=32,
    context_size=256
):

    start_time = time.perf_counter()

    image_t1_path = Path(image_t1_path)
    image_t2_path = Path(image_t2_path)
    output_dir = Path(output_dir)

    if not image_t1_path.exists():
        raise FileNotFoundError(
            f"T1 image not found: {image_t1_path}"
        )

    if not image_t2_path.exists():
        raise FileNotFoundError(
            f"T2 image not found: {image_t2_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    csdnet_dir = output_dir / "csdnet"
    regions_dir = output_dir / "regions"
    temporal_dir = output_dir / "remoteclip_temporal"
    semantic_dir = output_dir / "remoteclip_semantic"
    florence_dir = output_dir / "florence2"

    print("\n" + "=" * 60)
    print("GEOSPATIAL INVESTIGATION ENGINE")
    print("End-to-End Model Pipeline")
    print("=" * 60)

    print("\nT1:", image_t1_path)
    print("T2:", image_t2_path)
    print("Output:", output_dir)

    image_t1 = Image.open(image_t1_path).convert("RGB")
    image_t2 = Image.open(image_t2_path).convert("RGB")

    if image_t1.size != image_t2.size:
        raise ValueError(
            "T1 and T2 images must have identical dimensions."
        )

    image_width, image_height = image_t1.size

    print("\nImage dimensions:", image_width, "x", image_height)

    # --------------------------------------------------
    # STAGE 1: CSD-NET CHANGE DETECTION
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("STAGE 1: CSD-NET CHANGE DETECTION")
    print("=" * 60)

    stage_start = time.perf_counter()

    model, device = load_model()

    probability_map, prediction_mask = predict_change(
        model,
        device,
        image_t1,
        image_t2
    )

    save_results(
        image_t2,
        probability_map,
        prediction_mask,
        csdnet_dir
    )

    changed_pixels = int(prediction_mask.sum())
    total_pixels = int(prediction_mask.size)

    changed_percentage = round(
        changed_pixels / total_pixels * 100,
        4
    )

    del model
    release_gpu_memory()

    csdnet_time = round(
        time.perf_counter() - stage_start,
        3
    )

    print("\nCSD-Net completed.")
    print("Changed pixels:", changed_pixels)
    print("Changed area:", changed_percentage, "%")
    print("Processing time:", csdnet_time, "seconds")

    # --------------------------------------------------
    # STAGE 2: REGION EXTRACTION
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("STAGE 2: CHANGED REGION EXTRACTION")
    print("=" * 60)

    stage_start = time.perf_counter()

    mask_path = csdnet_dir / "change_mask.png"

    extract_regions(
        image_t1_path,
        image_t2_path,
        mask_path,
        regions_dir,
        min_area,
        padding
    )

    metadata_path = regions_dir / "regions.json"

    with open(metadata_path, "r", encoding="utf-8") as file:
        region_metadata = json.load(file)

    accepted_regions = region_metadata["accepted_regions"]

    region_time = round(
        time.perf_counter() - stage_start,
        3
    )

    print("\nRegion extraction completed.")
    print("Accepted regions:", accepted_regions)
    print("Processing time:", region_time, "seconds")

    # --------------------------------------------------
    # STAGE 3: REMOTECLIP TEMPORAL COMPARISON
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("STAGE 3: REMOTECLIP TEMPORAL COMPARISON")
    print("=" * 60)

    stage_start = time.perf_counter()

    temporal_analyzer = RemoteCLIPTemporalComparison(
        region_dir=regions_dir,
        output_dir=temporal_dir
    )

    temporal_results = (
        temporal_analyzer.compare_all_regions()
    )

    del temporal_analyzer
    release_gpu_memory()

    temporal_time = round(
        time.perf_counter() - stage_start,
        3
    )

    print("\nTemporal comparison completed.")
    print("Regions compared:", len(temporal_results))
    print("Processing time:", temporal_time, "seconds")

    # --------------------------------------------------
    # STAGE 4: REMOTECLIP SEMANTIC ANALYSIS
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("STAGE 4: REMOTECLIP SEMANTIC ANALYSIS")
    print("=" * 60)

    stage_start = time.perf_counter()

    semantic_analyzer = RemoteCLIPSemanticAnalysis(
        region_dir=regions_dir,
        output_dir=semantic_dir
    )

    semantic_results = (
        semantic_analyzer.analyze_all_regions()
    )

    del semantic_analyzer
    release_gpu_memory()

    semantic_time = round(
        time.perf_counter() - stage_start,
        3
    )

    print("\nSemantic analysis completed.")
    print("Regions analyzed:", len(semantic_results))
    print("Processing time:", semantic_time, "seconds")

    # --------------------------------------------------
    # STAGE 5: FLORENCE-2 REGION DESCRIPTIONS
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("STAGE 5: FLORENCE-2 REGION DESCRIPTIONS")
    print("=" * 60)

    stage_start = time.perf_counter()

    florence_results = analyze_regions(
        image_t1_path=image_t1_path,
        image_t2_path=image_t2_path,
        region_dir=regions_dir,
        output_dir=florence_dir,
        context_size=context_size
    )

    release_gpu_memory()

    florence_time = round(
        time.perf_counter() - stage_start,
        3
    )

    print("\nFlorence-2 analysis completed.")
    print("Processing time:", florence_time, "seconds")

    # --------------------------------------------------
    # STAGE 6: CONSOLIDATED REPORT
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("STAGE 6: GENERATING CONSOLIDATED REPORT")
    print("=" * 60)

    total_time = round(
        time.perf_counter() - start_time,
        3
    )

    report = {
        "project": "Geospatial Investigation Engine",

        "input": {
            "t1_image": image_t1_path.name,
            "t2_image": image_t2_path.name,
            "image_dimensions": [
                image_width,
                image_height
            ]
        },

        "change_detection": {
            "model": "Custom CSD-Net Base",
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,
            "changed_area_percentage": changed_percentage,
            "probability_map": str(
                csdnet_dir / "change_probability.png"
            ),
            "change_mask": str(mask_path),
            "change_overlay": str(
                csdnet_dir / "change_overlay.png"
            )
        },

        "region_extraction": {
            "minimum_area": min_area,
            "padding": padding,
            "total_components": region_metadata[
                "total_components"
            ],
            "accepted_regions": accepted_regions,
            "regions": region_metadata["regions"]
        },

        "remoteclip_temporal_analysis": {
            "description": (
                "Cosine similarity between T1 and T2 "
                "region image embeddings."
            ),
            "results": temporal_results
        },

        "remoteclip_semantic_analysis": {
            "description": (
                "Provisional semantic classification "
                "using multiple text prompts."
            ),
            "results": semantic_results
        },

        "florence2_region_analysis": {
            "description": (
                "Supplementary natural-language descriptions. "
                "Descriptions require visual verification."
            ),
            "results": florence_results
        },

        "processing_times_seconds": {
            "csdnet": csdnet_time,
            "region_extraction": region_time,
            "remoteclip_temporal": temporal_time,
            "remoteclip_semantic": semantic_time,
            "florence2": florence_time,
            "total": total_time
        },

        "reliability_notes": [
            "CSD-Net predictions are model-generated change evidence, "
            "not independently verified real-world changes.",

            "Connected components are candidate regions and may "
            "represent fragmented parts of one physical object.",

            "RemoteCLIP similarity measures semantic resemblance; "
            "it is not a probability of change.",

            "RemoteCLIP semantic categories are provisional predictions.",

            "Florence-2 descriptions are supplementary and may "
            "contain incorrect visual interpretations."
        ]
    }

    report_path = output_dir / "pipeline_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4
        )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("Image dimensions:", image_width, "x", image_height)
    print("Changed area:", changed_percentage, "%")
    print("Accepted regions:", accepted_regions)
    print("Temporal results:", len(temporal_results))
    print("Semantic results:", len(semantic_results))
    print("Florence-2 results:", (
        len(florence_results)
        if isinstance(florence_results, list)
        else "Report generated"
    ))

    print("\nTotal processing time:", total_time, "seconds")
    print("Consolidated report:", report_path)

    return report


def main():

    parser = argparse.ArgumentParser(
        description="Run the complete GEI model pipeline."
    )

    parser.add_argument(
        "--image-t1",
        required=True
    )

    parser.add_argument(
        "--image-t2",
        required=True
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR)
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=30
    )

    parser.add_argument(
        "--padding",
        type=int,
        default=32
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=256
    )

    args = parser.parse_args()

    run_pipeline(
        image_t1_path=args.image_t1,
        image_t2_path=args.image_t2,
        output_dir=Path(args.output),
        min_area=args.min_area,
        padding=args.padding,
        context_size=args.context_size
    )


if __name__ == "__main__":
    main()
