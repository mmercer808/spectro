import pytest
from engine.time.camera import TimeCamera, TimeCameraMode, TimeCameraConfig


def test_beat_to_px_basic():
    camera = TimeCamera(left_beat=0.0, window_beats=16.0)
    camera._panel_width_px = 800.0
    
    # Beat 0 is at pixel 0
    assert camera.beat_to_px(0.0) == 0.0
    
    # Beat 8 is at pixel 400 (halfway)
    assert camera.beat_to_px(8.0) == 400.0
    
    # Beat 16 is at pixel 800 (right edge)
    assert camera.beat_to_px(16.0) == 800.0

def test_px_to_beat_basic():
    camera = TimeCamera(left_beat=0.0, window_beats=16.0)
    camera._panel_width_px = 800.0
    
    assert camera.px_to_beat(0.0) == 0.0
    assert camera.px_to_beat(400.0) == 8.0
    assert camera.px_to_beat(800.0) == 16.0

def test_roundtrip():
    camera = TimeCamera(left_beat=4.0, window_beats=8.0)
    camera._panel_width_px = 400.0
    
    # Any beat should survive the round trip
    for beat in [4.0, 5.5, 8.0, 11.99]:
        px = camera.beat_to_px(beat)
        recovered = camera.px_to_beat(px)
        assert abs(recovered - beat) < 0.0001

def test_scrolled_view():
    camera = TimeCamera(left_beat=100.0, window_beats=16.0)
    camera._panel_width_px = 800.0
    
    # Beat 100 is now at pixel 0
    assert camera.beat_to_px(100.0) == 0.0
    # Beat 108 is at pixel 400
    assert camera.beat_to_px(108.0) == 400.0

def test_visibility():
    camera = TimeCamera(left_beat=10.0, window_beats=20.0)
    camera._panel_width_px = 800.0
    
    assert camera.is_beat_visible(10.0) == True   # Left edge
    assert camera.is_beat_visible(20.0) == True   # Middle
    assert camera.is_beat_visible(29.999) == True   # Just before right edge
    assert camera.is_beat_visible(30.0) == False   # Right edge
    assert camera.is_beat_visible(9.0) == False   # Before
    assert camera.is_beat_visible(31.0) == False  # After

if __name__ == "__main__": 
    test_beat_to_px_basic()
    test_px_to_beat_basic()
    test_roundtrip()
    test_scrolled_view()
    test_visibility()

