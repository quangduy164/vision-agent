def vision_analysis(image_path: str):
    confidence = 0.87  # demo
    heatmap_path = "outputs/heatmap.png"
    mask_path = "outputs/mask.png"

    return {
        "confidence": confidence,
        "heatmap": heatmap_path,
        "roi_mask": mask_path
    }
