from __future__ import annotations

from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SMARTDOC_ROOT = (
    PROJECT_ROOT
    / "data"
    / "smartdoc"
    / "sampleDataset"
)

INPUT_ROOT = (
    SMARTDOC_ROOT
    / "input_sample"
    / "background00"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "smartdoc_frames"
)

FRAMES_PER_VIDEO = 10


def extract_uniform_frames(
    video_path: Path,
    output_dir: Path,
    frames_per_video: int,
) -> None:
    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{video_path}"
        )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if total_frames <= 0:
        raise RuntimeError(
            f"Video contains no readable frames:\n"
            f"{video_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 80)
    print(f"VIDEO: {video_path.name}")
    print(f"TOTAL FRAMES: {total_frames}")
    print("=" * 80)

    if frames_per_video == 1:
        frame_indices = [0]
    else:
        frame_indices = [
            round(
                i
                * (total_frames - 1)
                / (frames_per_video - 1)
            )
            for i in range(
                frames_per_video
            )
        ]

    for output_index, frame_index in enumerate(
        frame_indices,
        start=1,
    ):
        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index,
        )

        success, frame = capture.read()

        if not success:
            print(
                f"[WARN] Could not read "
                f"frame {frame_index}"
            )
            continue

        output_path = (
            output_dir
            / (
                f"{video_path.stem}_"
                f"{output_index:02d}_"
                f"frame_{frame_index:05d}.png"
            )
        )

        cv2.imwrite(
            str(output_path),
            frame,
        )

        print(
            f"[SAVED] {output_path.name}"
        )

    capture.release()


def main() -> None:
    videos = sorted(
        INPUT_ROOT.glob("*.avi")
    )

    if not videos:
        raise FileNotFoundError(
            f"No AVI files found in:\n"
            f"{INPUT_ROOT}"
        )

    print("=" * 80)
    print("SMARTDOC FRAME EXTRACTION")
    print("=" * 80)
    print(f"Input directory : {INPUT_ROOT}")
    print(f"Output directory: {OUTPUT_ROOT}")
    print(f"Videos found    : {len(videos)}")
    print(
        f"Frames/video    : "
        f"{FRAMES_PER_VIDEO}"
    )

    for video_path in videos:
        output_dir = (
            OUTPUT_ROOT
            / video_path.stem
        )

        extract_uniform_frames(
            video_path=video_path,
            output_dir=output_dir,
            frames_per_video=FRAMES_PER_VIDEO,
        )

    print()
    print("=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()