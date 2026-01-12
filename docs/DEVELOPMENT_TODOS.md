# Audio Analyzer: Development TODOs

## Immediate Next Steps (Phase 1)

### 1. Project Setup
- [ ] Create project directory structure
- [ ] Set up Python virtual environment
- [ ] Install dependencies: `librosa`, `numpy`, `scipy`, `PySide6`, `moderngl`
- [ ] Create main.py entry point
- [ ] Basic window with ModernGL context

### 2. Audio Loading
- [ ] Load audio file (wav, mp3, flac)
- [ ] Get sample rate, duration, channels
- [ ] Convert to mono if stereo (or handle both)
- [ ] Store as numpy array for processing

### 3. Waveform Display
- [ ] Downsample waveform for display (can't draw every sample)
- [ ] Render as line or filled polygon
- [ ] Zoom and pan controls
- [ ] Playhead position indicator

### 4. Spectrogram Display
- [ ] Compute STFT (Short-Time Fourier Transform)
- [ ] Convert to dB scale for better visualization
- [ ] Render as texture in ModernGL
- [ ] Color mapping (grayscale? viridis? user choice?)
- [ ] Linear vs Log frequency toggle

### 5. Audio Playback
- [ ] Play audio (sounddevice or pygame.mixer or Qt multimedia)
- [ ] Sync playhead with playback position
- [ ] Play/Pause/Stop controls
- [ ] Seek by clicking on waveform

### 6. Basic Analysis Display
- [ ] BPM detection (librosa.beat.beat_track)
- [ ] Key detection (librosa.feature.chroma, then estimate)
- [ ] Display results in UI

---

## Phase 2: Structure Detection

- [ ] Phrase boundary detection
- [ ] Transient/onset detection
- [ ] Energy contour (RMS over time)
- [ ] Marker system (add numbered markers at points)
- [ ] Export/save marker data

---

## Phase 3: Track Comparison

- [ ] Load two tracks side by side
- [ ] Spectral complement visualization
- [ ] Time alignment tools
- [ ] Compatibility scoring algorithm

---

## Phase 4: Drum Slicer (Later)

- [ ] Onset detection for percussive audio
- [ ] Automatic slice point detection
- [ ] Slice rearrangement UI
- [ ] Export slices as individual files or new arrangement

---

## Technical Notes

### Spectrogram Math
```python
import librosa
import numpy as np

# Load audio
y, sr = librosa.load('track.mp3', sr=None)

# Compute spectrogram
D = librosa.stft(y, n_fft=2048, hop_length=512)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# S_db is now a 2D array: frequency bins × time frames
# Shape: (1025, num_frames) for n_fft=2048
```

### Log-Frequency Spectrogram
```python
# For log-frequency (constant-Q transform)
C = librosa.cqt(y, sr=sr)
C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)

# Or use mel spectrogram (perceptually spaced)
M = librosa.feature.melspectrogram(y=y, sr=sr)
M_db = librosa.power_to_db(M, ref=np.max)
```

### Linear vs Log Display Question
- Linear frequency: Low notes take up huge space, high notes compressed
- Log frequency: Each octave gets equal space (musical intervals equal)
- Mel scale: Perceptually uniform (how humans hear)

CloudyCadet asked about scaling piano to spectrogram — this is the linear case where low piano keys would be huge, high keys tiny. Log or mel scale makes the piano uniform.

### ModernGL Texture for Spectrogram
```python
import moderngl
import numpy as np

# Assuming ctx is your ModernGL context
# S_db is your spectrogram (2D numpy array)

# Normalize to 0-255
S_normalized = ((S_db - S_db.min()) / (S_db.max() - S_db.min()) * 255).astype(np.uint8)

# Create texture
texture = ctx.texture(S_normalized.shape[::-1], 1, S_normalized.tobytes())
texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
```

---

## Questions to Resolve

1. **Audio playback library?** 
   - `sounddevice` (simple, low latency)
   - `pygame.mixer` (game-oriented)
   - `Qt Multimedia` (integrated with PySide6)

2. **Start with ModernGL or plain Qt?**
   - ModernGL: Better for spectrogram (GPU texture), but more setup
   - Qt: QPainter is simpler, might be enough for prototype

3. **File format support?**
   - librosa handles most formats via soundfile/audioread
   - May need ffmpeg for some formats

---

## Reference: Key Libraries

```bash
pip install librosa          # Audio analysis
pip install numpy scipy      # Math
pip install PySide6          # GUI
pip install moderngl         # OpenGL
pip install sounddevice      # Audio playback
pip install madmom           # Beat/tempo detection (alternative to librosa)
```
