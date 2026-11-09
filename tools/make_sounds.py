"""Build sounds.bin (local voice clips) for the PalmStar ESP32 firmware.

Two steps, each a sub-command:

  split  One long recording per body part -> one WAV per line, for review.
         python make_sounds.py split <recordings dir> <clips dir>
         Every WAV in <recordings dir> is split on pauses (>= 0.6 s). The file
         name (without .wav) becomes the part name, e.g. neck.wav -> clips/neck/.

  pack   One folder per body part with one WAV per line -> sounds.bin.
         python make_sounds.py pack <clips dir> <sounds.bin>
         Folder names are the part names and must match the ESP32 sketch.

Clips are converted to 24 kHz mono 16-bit, peak normalized and faded; pack also
adds a short pause after each clip. Input must be WAV (16/24/32-bit PCM); export
MP3/M4A to WAV first, e.g. with Audacity.

sounds.bin layout (little endian), flashed to the "ffat" partition at 0x610000:
  b"PSND", u16 version=1, u16 clip count, u32 sample rate
  per clip: 8-byte part name (zero padded), u32 offset, u32 length in bytes
  then all clips as raw s16le PCM

Requires numpy (pip install numpy).
"""

import struct
import sys
import wave
from pathlib import Path

import numpy as np

RATE = 24000  # must match SPK_RATE in the ESP32 sketch
PARTITION_BYTES = 0x9E0000  # "16M Flash (3MB APP/9.9MB FATFS)" data partition

# split: pause detection
FRAME = 0.02  # seconds per analysis frame
VOICE_DB = -42  # a frame louder than this counts as speech
TAIL_DB = -58  # clip edges are extended while still louder than this
MIN_GAP = 0.6  # silence that separates two lines, seconds
MIN_LEN = 0.15  # shorter sounds are clicks or breaths, dropped
MAX_EXTEND = 0.6  # longest edge extension, seconds
PAD_START, PAD_END = 0.10, 0.30  # silence kept around a split line

FADE_IN, FADE_OUT = 0.01, 0.05  # seconds, avoids clicks at clip edges
PACK_TAIL = 0.20  # pause appended to every clip in pack, seconds


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a PCM WAV file as mono float samples in [-1, 1].

    Args:
        path: WAV file.

    Returns:
        (samples, sample_rate).

    Raises:
        SystemExit: If the file is not a supported PCM WAV.
    """
    try:
        with wave.open(str(path)) as w:
            ch, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
            raw = w.readframes(w.getnframes())
    except (wave.Error, EOFError) as e:
        sys.exit(f"读不了 {path}：{e}。请导出成 16 位 PCM 的 WAV 文件。")
    if width == 2:
        a = np.frombuffer(raw, "<i2").astype(np.float64) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        a = (b[:, 0].astype(np.int32) | b[:, 1].astype(np.int32) << 8
             | b[:, 2].astype(np.int8).astype(np.int32) << 16) / 8388608.0  # fmt: skip
    elif width == 4:
        a = np.frombuffer(raw, "<i4").astype(np.float64) / 2147483648.0
    else:
        sys.exit(f"{path} 是 {width * 8} 位的，不支持，请导出成 16 位 WAV。")
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, rate


def finish(clip: np.ndarray, rate: int, tail: float = 0.0) -> bytes:
    """Fade, resample to RATE, normalize and convert a clip to s16le bytes.

    Args:
        clip: Mono float samples.
        rate: Sample rate of ``clip``.
        tail: Seconds of silence to append.

    Returns:
        Raw 16-bit PCM at RATE.
    """
    clip = clip.copy()
    fi = min(int(FADE_IN * rate), len(clip))
    fo = min(int(FADE_OUT * rate), len(clip))
    clip[:fi] *= np.linspace(0, 1, fi)
    clip[len(clip) - fo :] *= np.linspace(1, 0, fo)
    if rate != RATE:
        # FFT resampling: drops everything above the new Nyquist, no aliasing
        n_out = max(1, int(round(len(clip) * RATE / rate)))
        spec = np.fft.rfft(clip)[: n_out // 2 + 1]
        clip = np.fft.irfft(spec, n_out) * (n_out / len(clip))
    clip *= 0.89 / max(np.abs(clip).max(), 1e-9)  # peak at about -1 dBFS
    clip = np.concatenate([clip, np.zeros(int(tail * RATE))])
    return (clip * 32767).astype("<i2").tobytes()


def split(src: Path, out: Path) -> None:
    """Split each long recording in ``src`` into one WAV per line under ``out``."""
    files = sorted(src.glob("*.wav"))
    if not files:
        sys.exit(f"{src} 里没有 WAV 文件。")
    for f in files:
        a, rate = load_wav(f)
        fl = int(rate * FRAME)
        frames = a[: len(a) // fl * fl].reshape(-1, fl)
        level = 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-9)

        segs, start, gap = [], None, 0
        for i, voiced in enumerate(level > VOICE_DB):
            if voiced:
                start = i if start is None else start
                gap = 0
            elif start is not None:
                gap += 1
                if gap * FRAME >= MIN_GAP:
                    segs.append((start, i - gap + 1))
                    start, gap = None, 0
        if start is not None:
            segs.append((start, len(level)))
        segs = [(s, e) for s, e in segs if (e - s) * FRAME >= MIN_LEN]

        part_dir = out / f.stem
        part_dir.mkdir(parents=True, exist_ok=True)
        for old in part_dir.glob("*.wav"):
            old.unlink()
        lens = []
        limit = int(MAX_EXTEND / FRAME)
        for idx, (s, e) in enumerate(segs, 1):
            # Extend edges so soft word endings are not cut off.
            for _ in range(limit):
                if s > 0 and level[s - 1] > TAIL_DB:
                    s -= 1
            for _ in range(limit):
                if e < len(level) and level[e] > TAIL_DB:
                    e += 1
            lo = max(0, int((s * FRAME - PAD_START) * rate))
            hi = min(len(a), int((e * FRAME + PAD_END) * rate))
            pcm = finish(a[lo:hi], rate)
            with wave.open(str(part_dir / f"{idx:02d}.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                w.writeframes(pcm)
            lens.append(round(len(pcm) / 2 / RATE, 1))
        print(f"{f.stem}: {len(lens)} 句，{sum(lens):.1f} 秒，每句长度 {lens}")
    print(
        f"\n切好了，放在 {out}。请逐句试听：删掉不要的，需要合并的用 Audacity 拼起来，"
    )
    print("文件夹名改成英文部位名（和 ESP32 程序一致），然后运行 pack 打包。")


def pack(src: Path, dest: Path) -> None:
    """Pack every ``src/<part>/*.wav`` into a sounds.bin file at ``dest``."""
    entries: list[tuple[str, bytes]] = []
    for part_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        name = part_dir.name
        if not name.isascii() or not 1 <= len(name) <= 8 or " " in name:
            sys.exit(
                f"文件夹名 “{name}” 不行：要用 1~8 个英文字母或数字，例如 neck、hand。"
            )
        clips = sorted(part_dir.glob("*.wav"))
        for f in clips:
            a, rate = load_wav(f)
            entries.append((name, finish(a, rate, tail=PACK_TAIL)))
        total = sum(len(p) for n, p in entries if n == name) / 2 / RATE
        print(f"{name}: {len(clips)} 句，{total:.1f} 秒")
    if not entries:
        sys.exit(f"{src} 里没有找到 <部位名>/*.wav。")

    header_size = 12 + 16 * len(entries)
    table, offset = b"", header_size
    for name, pcm in entries:
        table += struct.pack("<8sII", name.encode(), offset, len(pcm))
        offset += len(pcm)
    data = (
        b"PSND"
        + struct.pack("<HHI", 1, len(entries), RATE)
        + table
        + b"".join(p for _, p in entries)
    )
    if len(data) > PARTITION_BYTES:
        sys.exit(
            f"总共 {len(data) / 1048576:.2f} MB，超过了 {PARTITION_BYTES / 1048576:.2f} MB "
            "的存储空间，请删掉一些或缩短录音。"
        )
    dest.write_bytes(data)
    print(f"\n已生成 {dest}：{len(entries)} 句，{len(data) / 1048576:.2f} MB "
          f"（最多 {PARTITION_BYTES / 1048576:.2f} MB）")  # fmt: skip


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("split", "pack"):
        print(__doc__)
        sys.exit(1)
    (split if sys.argv[1] == "split" else pack)(Path(sys.argv[2]), Path(sys.argv[3]))
