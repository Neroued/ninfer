from __future__ import annotations

import pytest

from tools.convert.qwen3_8_27b import convert_nvfp4


def test_converter_rejects_wrong_basename_before_reading_sources(tmp_path) -> None:
    output = tmp_path / "qwen3_8_27b.ninfer"
    with pytest.raises(ValueError, match="output basename"):
        convert_nvfp4.convert(
            tmp_path / "missing-official",
            tmp_path / "missing-quantized",
            output,
            device="cpu",
        )
    assert not output.exists()
