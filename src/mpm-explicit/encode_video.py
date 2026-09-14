"""Render an mpm-explicit Rerun recording (.rrd) to an MP4 video.

This started life as Rerun's "video stream" example, which shows how to
*encode* synthetic frames with PyAV and log them into Rerun as a
`rr.VideoStream`. That's the opposite direction of what we want here, so
this version instead:

1. Reads an existing `.rrd` recording back with `rerun.experimental.RrdReader`.
2. Projects the `mpm/particles` point cloud (and the static bounding box) to
   2D with a small vectorized camera/rasterizer -- no GPU or rerun viewer
   required.
3. Muxes the resulting frames straight into an `.mp4` file with PyAV,
   instead of streaming an elementary stream back into Rerun.

Usage
-----
    uv run encode_video.py recordings/run_20260912_104022.rrd
    uv run encode_video.py recordings/run_20260912_104022.rrd -o out.mp4 --orbit 360

Run `uv run encode_video.py --help` for all options.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw
from rerun.experimental import RrdReader
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

ENTITY_PARTICLES = "mpm/particles"
ENTITY_BOX = "mpm/static/box"

CODECS = {"h264": "libx264", "h265": "libx265"}


def unpack_colors(packed: np.ndarray) -> np.ndarray:
    """Unpack Rerun's 0xRRGGBBAA `Color` components into an (N, 3) uint8 RGB array."""
    return packed.astype(">u4").view(np.uint8).reshape(-1, 4)[:, :3]


def read_particle_frames(reader: RrdReader, stride: int) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Read every logged `mpm/particles` frame as (time_seconds, positions, colors_rgb)."""
    frames = []
    for chunk in reader.stream().filter(content=ENTITY_PARTICLES).to_chunks():
        batch = chunk.to_record_batch()
        t = batch.column("step")[0].value / 1e9

        positions = np.asarray(batch.column("Points3D:positions").values.values, dtype=np.float32).reshape(-1, 3)
        colors = unpack_colors(np.asarray(batch.column("Points3D:colors").values))

        # A few frames contain NaN positions (e.g. particles gone unstable /
        # out of domain); drop them rather than let them corrupt the render.
        finite = np.isfinite(positions).all(axis=1)
        if not finite.all():
            positions, colors = positions[finite], colors[finite]

        if stride > 1:
            positions = positions[::stride]
            colors = colors[::stride]

        frames.append((t, positions, colors))

    frames.sort(key=lambda f: f[0])
    return frames


def read_box(reader: RrdReader) -> tuple[np.ndarray, np.ndarray] | None:
    """Read the static `mpm/static/box` center/half-size, if present."""
    chunks = reader.stream().filter(content=ENTITY_BOX).to_chunks()
    if not chunks:
        return None
    batch = chunks[0].to_record_batch()
    # Each row of a Boxes3D column is itself a list (one entry per box); this
    # project only ever logs a single box, so take the first one.
    center = np.asarray(batch.column("Boxes3D:centers")[0].as_py()[0], dtype=np.float32)
    half_size = np.asarray(batch.column("Boxes3D:half_sizes")[0].as_py()[0], dtype=np.float32)
    return center, half_size


def box_corners(center: np.ndarray, half_size: np.ndarray) -> np.ndarray:
    signs = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float32)
    return center + signs * half_size


BOX_EDGES = [
    (0, 1), (0, 2), (0, 4), (3, 1), (3, 2), (3, 7),
    (5, 1), (5, 4), (5, 7), (6, 2), (6, 4), (6, 7),
]


class Camera:
    """A simple pinhole camera that looks at a fixed target from an orbiting position."""

    def __init__(self, target: np.ndarray, distance: float, elevation_deg: float, width: int, height: int, fov_deg: float = 50.0):
        self.target = target
        self.distance = distance
        self.elevation = np.radians(elevation_deg)
        self.width = width
        self.height = height
        self.focal = (height / 2) / np.tan(np.radians(fov_deg) / 2)

    def basis(self, azimuth_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        az = np.radians(azimuth_deg)
        offset = self.distance * np.array([
            np.cos(self.elevation) * np.sin(az),
            np.cos(self.elevation) * np.cos(az),
            np.sin(self.elevation),
        ])
        eye = self.target + offset
        forward = (self.target - eye) / np.linalg.norm(self.target - eye)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        return eye, right, up, forward

    def project(self, points: np.ndarray, azimuth_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project world points to (pixel_xy, depth, in_front_mask)."""
        eye, right, up, forward = self.basis(azimuth_deg)
        rel = points - eye
        cx = rel @ right
        cy = rel @ up
        cz = rel @ forward

        in_front = cz > 1e-3
        cz_safe = np.where(in_front, cz, 1.0)
        px = (cx / cz_safe) * self.focal + self.width / 2
        py = -(cy / cz_safe) * self.focal + self.height / 2
        return np.stack([px, py], axis=-1), cz, in_front


def render_frame(
    camera: Camera,
    azimuth_deg: float,
    positions: np.ndarray,
    colors: np.ndarray,
    box: tuple[np.ndarray, np.ndarray] | None,
    background: tuple[int, int, int],
    point_radius: int,
) -> np.ndarray:
    width, height = camera.width, camera.height
    screen, depth, in_front = camera.project(positions, azimuth_deg)
    px, py = screen[:, 0], screen[:, 1]

    # Compare against bounds as floats *before* casting to int32: a handful of
    # particles in these recordings have blown up to astronomically large
    # (but finite) coordinates, and casting those directly would overflow.
    in_bounds = (
        in_front
        & np.isfinite(px)
        & np.isfinite(py)
        & (px >= -point_radius)
        & (px < width + point_radius)
        & (py >= -point_radius)
        & (py < height + point_radius)
    )
    ix = px[in_bounds].astype(np.int32)
    iy = py[in_bounds].astype(np.int32)
    depth, colors = depth[in_bounds], colors[in_bounds]

    # Farthest-first so nearer particles are the last (winning) write at each pixel.
    order = np.argsort(-depth)
    ix, iy, colors = ix[order], iy[order], colors[order]

    if point_radius > 1:
        offsets = np.array(
            [(dy, dx) for dy in range(-point_radius + 1, point_radius) for dx in range(-point_radius + 1, point_radius)]
        )
        ix = (ix[:, None] + offsets[None, :, 1]).ravel()
        iy = (iy[:, None] + offsets[None, :, 0]).ravel()
        colors = np.repeat(colors, len(offsets), axis=0)

    in_bounds = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    ix, iy, colors = ix[in_bounds], iy[in_bounds], colors[in_bounds]

    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = background
    image[iy, ix] = colors

    if box is not None:
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        corners, _, _ = camera.project(box_corners(*box), azimuth_deg)
        for a, b in BOX_EDGES:
            xy = [(float(corners[a][0]), float(corners[a][1])), (float(corners[b][0]), float(corners[b][1]))]
            draw.line(xy, fill=(120, 120, 130), width=1)
        image = np.asarray(pil_image)

    return image


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rrd", type=Path, help="Path to the input .rrd recording.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .mp4 path (default: alongside the .rrd file).")
    parser.add_argument("--fps", type=float, default=60.0, help="Output video frame rate (default: 60, matching the sim's render rate).")
    parser.add_argument("--width", type=int, default=1280, help="Video width in pixels.")
    parser.add_argument("--height", type=int, default=960, help="Video height in pixels.")
    parser.add_argument("--stride", type=int, default=1, help="Render only every Nth particle (useful for a fast preview).")
    parser.add_argument("--point-size", type=int, default=1, help="Point splat radius in pixels.")
    parser.add_argument("--azimuth", type=float, default=90.0, help="Starting camera azimuth, in degrees.")
    parser.add_argument("--elevation", type=float, default=15.0, help="Camera elevation angle, in degrees.")
    parser.add_argument("--orbit", type=float, default=0.0, help="Total azimuth rotation over the whole video, in degrees (0 = static camera).")
    parser.add_argument("--distance-factor", type=float, default=1.5, help="Camera distance, as a multiple of the scene's bounding radius.")
    parser.add_argument("--background", type=str, default="#12121a", help="Background color, as a hex string.")
    parser.add_argument("--codec", choices=CODECS, default="h264", help="Video codec.")
    args = parser.parse_args()

    output = args.output or args.rrd.with_suffix(".mp4")
    background = hex_to_rgb(args.background)

    print(f"Reading {args.rrd} ...")
    reader = RrdReader(args.rrd)
    frames = read_particle_frames(reader, args.stride)
    if not frames:
        raise SystemExit(f"No '{ENTITY_PARTICLES}' data found in {args.rrd}")
    box = read_box(reader)

    if box is not None:
        center, half_size = box
    else:
        # No static box logged: fall back to the first frame's particle bounds.
        first_positions = frames[0][1]
        center = (first_positions.min(0) + first_positions.max(0)) / 2
        half_size = (first_positions.max(0) - first_positions.min(0)) / 2

    distance = float(np.linalg.norm(half_size)) * args.distance_factor
    camera = Camera(center, distance, args.elevation, args.width, args.height)

    container = av.open(str(output), mode="w")
    stream = container.add_stream(CODECS[args.codec], rate=round(args.fps))
    stream.width = args.width
    stream.height = args.height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "20", "preset": "medium"}

    print(f"Rendering {len(frames)} frames to {output} ...")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        "•",
        MofNCompleteColumn(),
        "•",
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Frames", total=len(frames))
        for i, (_t, positions, colors) in enumerate(frames):
            azimuth = args.azimuth + args.orbit * (i / max(len(frames) - 1, 1))
            image = render_frame(camera, azimuth, positions, colors, box, background, args.point_size)

            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

            progress.update(task, advance=1)

    for packet in stream.encode():
        container.mux(packet)
    container.close()

    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
