from __future__ import annotations

"""
3x3 transparent isometric cube matrix renderer
- Transparent RGBA background
- Reference-like perspective / cube arrangement / shadows / guide lines
- Value in [0, 1] controls BOTH:
    1) cube color
    2) cube height

Dependencies:
    pip install numpy pillow

Run:
    python cube_matrix_reference_height.py
"""

from pathlib import Path
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


# ---------------------------------------------------------------------
# Geometry calibrated to the supplied reference image.
# Row 0 = back row, row 2 = front row.
# ---------------------------------------------------------------------
BASE_W, BASE_H = 254, 186

BASE_CENTERS = np.array(
    [
        [[90.0, 21.0], [146.0, 31.0], [202.0, 69.0]],
        [[64.0, 51.0], [113.0, 59.0], [168.0, 91.0]],
        [[39.0, 91.0], [87.0, 106.0], [135.0, 120.0]],
    ],
    dtype=float,
)

# Projected top-plane axes
BASE_U = np.array([40.0, 11.0], dtype=float)   # left -> right edge on top face
BASE_V = np.array([27.0, -21.0], dtype=float)  # depth edge on top face

# Maximum vertical drop for a full-height cube (value = 1.0)
BASE_HV = np.array([0.0, 42.0], dtype=float)

TEXT_CLOCKWISE_DEG = 15.4
BASE_FONT_SIZE = 18.0
BASE_EDGE_WIDTH = 1.30

BASE_GUIDE_SEGMENTS = (
    ((122.0, 251.0), (-80.0, 49.0)),
    ((122.0, 251.0), (316.0, 0.0)),
    ((240.0, 98.0), (310.0, 98.0)),
)

# Continuous palette: blue -> green -> amber -> red
PALETTE_STOPS = (
    (0.00, (78, 128, 184)),
    (0.30, (166, 207, 181)),
    (0.62, (246, 190, 107)),
    (1.00, (207, 76, 64)),
)


def _mix(c1, c2, t: float):
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(
        int(round((1.0 - t) * c1[i] + t * c2[i]))
        for i in range(3)
    )


def _value_to_rgb(value: float, palette_stops=PALETTE_STOPS):
    v = float(np.clip(value, 0.0, 1.0))
    stops = tuple(sorted(palette_stops, key=lambda x: x[0]))

    if v <= stops[0][0]:
        return tuple(stops[0][1])
    if v >= stops[-1][0]:
        return tuple(stops[-1][1])

    for (x0, c0), (x1, c1) in zip(stops[:-1], stops[1:]):
        if x0 <= v <= x1:
            t = 0.0 if x1 == x0 else (v - x0) / (x1 - x0)
            return _mix(c0, c1, t)

    return tuple(stops[-1][1])


def _face_colors(value01: float, palette_stops=PALETTE_STOPS):
    """
    Lighting tuned to the reference:
    top = slightly lighter, front = near base, right = darker.
    """
    base = _value_to_rgb(value01, palette_stops)
    top = _mix(base, (255, 255, 255), 0.08)
    front = _mix(base, (0, 0, 0), 0.02)
    right = _mix(base, (0, 0, 0), 0.17)
    return top, front, right


def _find_font(size_px: int):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    )
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size_px)

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size_px)
    except OSError:
        return ImageFont.load_default()


def _draw_closed_polygon(draw, points, fill, outline, width):
    pts = [tuple(int(round(v)) for v in p) for p in points]
    draw.polygon(pts, fill=fill)
    draw.line(pts + [pts[0]], fill=outline, width=width, joint="curve")


def _make_rotated_text(text: str, font, angle_deg: float):
    probe = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    d = ImageDraw.Draw(probe)
    box = d.textbbox((0, 0), text, font=font)

    tw = max(1, box[2] - box[0])
    th = max(1, box[3] - box[1])
    pad = 6

    sprite = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sprite)
    sd.text(
        (pad - box[0], pad - box[1]),
        text,
        font=font,
        fill=(8, 8, 8, 255),
    )

    return sprite.rotate(
        -angle_deg,  # clockwise
        expand=True,
        resample=Image.Resampling.BICUBIC,
    )


def render_cube_matrix(
    values,
    output_path: str | Path = "cube_matrix.png",
    *,
    value_range=(0.0, 1.0),
    auto_range: bool = False,
    scale: float = 1.0,
    supersample: int = 4,
    label_format: str = "{:.1f}",
    show_floor_guides: bool = True,
    show_shadows: bool = True,
    palette_stops=PALETTE_STOPS,
    min_height_ratio: float = 0.0,
):
    """
    Render a 3x3 transparent PNG.

    IMPORTANT HEIGHT RULE:
        value = 1.0 -> 100% cube height
        value = 0.6 -> 60% cube height
        value = 0.0 -> 0% cube height (or min_height_ratio if set > 0)

    Parameters
    ----------
    values : 3x3 array-like
        Row 0 is the back row, row 2 is the front row.

    value_range : tuple
        Input numeric range mapped to [0, 1] when auto_range=False.

    auto_range : bool
        If True, normalize by the min/max of the input array.
        If False, use `value_range`.

    min_height_ratio : float
        Extra minimum visible height in [0,1].
        Example:
            0.0  -> zero really means zero height
            0.15 -> even value=0 still shows 15% height
    """
    values = np.asarray(values, dtype=float)
    if values.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 array, got shape {values.shape}.")

    if scale <= 0:
        raise ValueError("scale must be > 0.")
    if supersample < 1:
        raise ValueError("supersample must be >= 1.")
    if not (0.0 <= min_height_ratio <= 1.0):
        raise ValueError("min_height_ratio must be in [0, 1].")

    if auto_range:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            raise ValueError("values contain no finite numeric range.")
        if vmax == vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = map(float, value_range)
        if vmax == vmin:
            raise ValueError("value_range endpoints must be different.")

    def norm(v):
        return float(np.clip((float(v) - vmin) / (vmax - vmin), 0.0, 1.0))

    out_w = int(round(BASE_W * scale))
    out_h = int(round(BASE_H * scale))
    S = int(supersample)
    geom = float(scale * S)

    canvas = Image.new("RGBA", (out_w * S, out_h * S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    centers = BASE_CENTERS * geom
    u = BASE_U * geom
    v = BASE_V * geom
    hv_full = BASE_HV * geom

    # -----------------------------------------------------------------
    # Floor / guide lines
    # -----------------------------------------------------------------
    if show_floor_guides:
        guide = (82, 82, 82, 145)
        guide_width = max(1, int(round(1.15 * geom)))
        for p0, p1 in BASE_GUIDE_SEGMENTS:
            a = tuple(int(round(x * geom)) for x in p0)
            b = tuple(int(round(x * geom)) for x in p1)
            draw.line([a, b], fill=guide, width=guide_width)

    # -----------------------------------------------------------------
    # Soft contact shadows
    # -----------------------------------------------------------------
    if show_shadows:
        shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow)
        shadow_offset = np.array([2.5, 4.0]) * geom

        for r in range(3):
            for c in range(3):
                # Ground footprint is fixed, independent of height
                ground_center = centers[r, c] + hv_full

                A0 = ground_center - (u + v) / 2.0
                B0 = A0 + u
                C0 = B0 + v
                D0 = A0 + v

                poly = [
                    A0 + shadow_offset,
                    B0 + shadow_offset,
                    C0 + shadow_offset,
                    D0 + shadow_offset,
                ]
                poly = [tuple(int(round(q)) for q in p) for p in poly]
                sd.polygon(poly, fill=(25, 25, 25, 50))

        shadow = shadow.filter(
            ImageFilter.GaussianBlur(radius=max(1.0, 3.0 * geom))
        )
        canvas = Image.alpha_composite(canvas, shadow)
        draw = ImageDraw.Draw(canvas)

    # -----------------------------------------------------------------
    # Cubes / columns
    # -----------------------------------------------------------------
    edge = (43, 43, 43, 235)
    edge_width = max(1, int(round(BASE_EDGE_WIDTH * geom)))

    for r in range(3):
        for c in range(3):
            value01 = norm(values[r, c])

            # value controls height
            # 1.0 -> full height, 0.6 -> 60% height
            height_ratio = min_height_ratio + value01 * (1.0 - min_height_ratio)
            cube_hv = hv_full * height_ratio

            # Keep all columns on the same floor; grow upward as value increases
            ground_center = centers[r, c] + hv_full
            top_center = ground_center - cube_hv

            A = top_center - (u + v) / 2.0
            B = A + u
            C = B + v
            D = A + v

            top_col, front_col, right_col = _face_colors(value01, palette_stops)

            # If the height is 0, only a degenerate top face would remain.
            # We skip nearly-zero values for cleaner output.
            if np.linalg.norm(cube_hv) < 1e-6:
                continue

            front_face = [A, B, B + cube_hv, A + cube_hv]
            right_face = [B, C, C + cube_hv, B + cube_hv]
            top_face = [A, B, C, D]

            _draw_closed_polygon(
                draw,
                front_face,
                (*front_col, 255),
                edge,
                edge_width,
            )
            _draw_closed_polygon(
                draw,
                right_face,
                (*right_col, 255),
                edge,
                edge_width,
            )
            _draw_closed_polygon(
                draw,
                top_face,
                (*top_col, 255),
                edge,
                edge_width,
            )

    # -----------------------------------------------------------------
    # Labels on top faces
    # -----------------------------------------------------------------
    font = _find_font(max(1, int(round(BASE_FONT_SIZE * geom))))

    for r in range(3):
        for c in range(3):
            value01 = norm(values[r, c])
            height_ratio = min_height_ratio + value01 * (1.0 - min_height_ratio)
            cube_hv = hv_full * height_ratio

            if np.linalg.norm(cube_hv) < 1e-6:
                continue

            ground_center = centers[r, c] + hv_full
            top_center = ground_center - cube_hv

            text = label_format.format(float(values[r, c]))
            sprite = _make_rotated_text(
                text,
                font,
                angle_deg=TEXT_CLOCKWISE_DEG,
            )

            cx, cy = top_center
            pos = (
                int(round(cx - sprite.width / 2.0)),
                int(round(cy - sprite.height / 2.0)),
            )
            canvas.alpha_composite(sprite, dest=pos)

    if S != 1:
        canvas = canvas.resize((out_w, out_h), Image.Resampling.LANCZOS)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, "PNG")
    return output_path


if __name__ == "__main__":
    # Example:
    # 0.6 means:
    #   - color mapped from 0.6
    #   - height = 60% of full height
    # DEMO_VALUES = np.array(
    #     [
    #         [0.00, 0.30, 0.60],
    #         [0.20, 1.00, 0.80],
    #         [0.10, 0.50, 0.70],
    #     ],
    #     dtype=float,
    # )
    DEMO_VALUES = np.array(
        [
            [0.40, 0.40, 0.40],
            [0.40, 0.40, 0.40],
            [0.40, 0.40, 0.40],
        ],
        dtype=float,
    )

    out = render_cube_matrix(
        DEMO_VALUES,
        "cube_matrix.png",
        value_range=(0.0, 1.0),
        auto_range=False,
        scale=1.0,
        supersample=4,
        label_format="{:.1f}",
        show_floor_guides=True,
        show_shadows=True,
        min_height_ratio=0.0,   # change to 0.15 if you want 0-value cubes still visible
    )
    print(f"Saved: {out.resolve()}")
