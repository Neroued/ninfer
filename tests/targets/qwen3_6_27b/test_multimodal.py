from __future__ import annotations

import torch

from tools.reference.qwen3_6_27b_rtx5090.multimodal import (
    MultimodalBatch,
    build_mrope_positions,
)


def test_mrope_and_chunked_visual_embedding_alignment():
    # Two image items and a two-frame video.  Timestamp text separates the
    # video frames exactly as the library processor emits them.
    types = torch.tensor(
        [0, 1, 1, 1, 1, 0, 2, 2, 0, 2, 2, 0, 1, 0], dtype=torch.long
    )
    image_grid = torch.tensor([[1, 4, 4], [1, 2, 2]], dtype=torch.long)
    video_grid = torch.tensor([[2, 2, 4]], dtype=torch.long)
    positions, delta = build_mrope_positions(types, image_grid, video_grid)

    assert positions.tolist() == [
        [0, 1, 1, 1, 1, 3, 4, 4, 6, 7, 7, 9, 10, 11],
        [0, 1, 1, 2, 2, 3, 4, 4, 6, 7, 7, 9, 10, 11],
        [0, 1, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ]
    assert delta == -2

    batch = MultimodalBatch(
        input_ids=torch.arange(types.numel()),
        mm_token_type_ids=types,
        position_ids=positions,
        rope_delta=delta,
        pixel_values=None,
        image_grid_thw=image_grid,
        pixel_values_videos=None,
        video_grid_thw=video_grid,
    )
    image = torch.tensor([[10.0], [11.0], [12.0], [13.0], [20.0]])
    video = torch.tensor([[30.0], [31.0], [40.0], [41.0]])
    first = torch.arange(7, dtype=torch.float32).unsqueeze(1) + 100
    second = torch.arange(7, 14, dtype=torch.float32).unsqueeze(1) + 100
    batch.scatter_visual_embeddings_(first, image, video, offset=0)
    batch.scatter_visual_embeddings_(second, image, video, offset=7)

    assert torch.cat((first, second)).flatten().tolist() == [
        100.0,
        10.0,
        11.0,
        12.0,
        13.0,
        105.0,
        30.0,
        31.0,
        108.0,
        40.0,
        41.0,
        111.0,
        20.0,
        113.0,
    ]
