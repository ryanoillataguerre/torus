import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import imageio.v2 as imageio
from moviepy.editor import ImageSequenceClip
import os
import cProfile, pstats, io
import concurrent.futures
import tqdm

import matplotlib

matplotlib.use("Agg")  # Select non‑interactive backend just once, up front

# --- Worker‑process globals (populated once per worker) --------------------
_u = _v = _texture = None
_R = _r = _tex_w = _tex_h = _polar_ratio = None
_view_elev = _view_azim = None


def _init_worker(u, v, texture, R, r, tex_w, tex_h, polar_ratio, view_elev, view_azim):
    """
    Cache large, read‑only arrays in each worker so we don’t have to
    pickle & ship them for every single frame.
    """
    global _u, _v, _texture, _R, _r, _tex_w, _tex_h, _polar_ratio
    global _view_elev, _view_azim
    _u = u
    _v = v
    _texture = texture
    _R = R
    _r = r
    _tex_w = tex_w
    _tex_h = tex_h
    _polar_ratio = polar_ratio
    _view_elev = view_elev
    _view_azim = view_azim


def create_frame(params):
    """Generate a single frame of the animation."""
    phi, index = params

    # pull cached globals
    u, v, texture = _u, _v, _texture
    tex_w, tex_h = _tex_w, _tex_h
    R, r = _R, _r
    polar_ratio = _polar_ratio
    view_elev, view_azim = _view_elev, _view_azim

    # Swirl parameters
    u_offset = (u / (2 * np.pi) + phi) % 1.0  # toroidal (around big ring)
    v_offset = (v / (2 * np.pi) + phi * polar_ratio) % 1.0  # poloidal (around tube)

    u_idx = (u_offset * tex_w).astype(int)
    v_idx = (v_offset * tex_h).astype(int)

    facecolors = texture[v_idx, u_idx]
    facecolors = np.concatenate(
        [facecolors, np.ones((*facecolors.shape[:2], 1))], axis=-1
    )

    # Calculate torus coordinates
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)

    # Create the plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        x,
        y,
        z,
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    ax.view_init(elev=view_elev, azim=view_azim)
    ax.set_axis_off()
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_zlim(-6, 6)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # Render to buffer
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    frame = imageio.imread(buf)
    buf.close()

    return index, frame


def create_torus_animation(
    texture_path="space_8.jpeg",
    output_path="torus_spacetime_animation_space.mp4",
    R=3.0,
    r=2.0,
    u_res=480,
    v_res=300,
    speed=2.0,
    base_frames=360,
    toroidal_cycles=1,
    polar_ratio=3.0,
    frame_interval=0.03,
    view_elev=45,
    view_azim=45,
    max_workers=None,  # Default to CPU count
):
    """
    Create a torus animation with a space texture wrapped around it.

    Args:
        texture_path: Path to the texture image
        output_path: Path for the output video
        R: Major radius of the torus
        r: Minor radius of the torus
        u_res, v_res: Resolution parameters for the torus
        speed: Animation speed
        base_frames: Base number of frames
        toroidal_cycles: Number of toroidal cycles
        polar_ratio: Ratio for polar rotation
        frame_interval: Interval between frames
        view_elev: Elevation angle for viewing
        view_azim: Azimuth angle for viewing
        max_workers: Maximum number of parallel workers
    """
    # Load your space image texture
    texture_img = Image.open(texture_path).convert("RGB")
    texture = np.asarray(texture_img) / 255.0

    tex_h, tex_w, _ = texture.shape

    # Torus geometry
    u = np.linspace(0, 2 * np.pi, u_res)
    v = np.linspace(0, 2 * np.pi, v_res)
    u, v = np.meshgrid(u, v)

    # Number of frames calculation
    n_frames = int(round(base_frames / speed))
    phis = np.linspace(0.0, toroidal_cycles, n_frames + 1, endpoint=True)

    # Create parameter list for each frame
    params_list = [(phi, i) for i, phi in enumerate(phis[:-1])]

    # Create a dictionary to store the frames
    frames_dict = {}

    # Use ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(u, v, texture, R, r, tex_w, tex_h, polar_ratio, view_elev, view_azim),
    ) as executor:
        # Submit all frame generation tasks and show progress
        future_to_index = {
            executor.submit(create_frame, params): params[1] for params in params_list
        }

        # Process results as they complete with a progress bar
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(future_to_index),
            total=len(params_list),
            desc="Rendering frames",
        ):
            index, frame = future.result()
            frames_dict[index] = frame

    # Sort frames by index
    frames = [frames_dict[i] for i in range(n_frames)]

    # Output as .mp4
    clip = ImageSequenceClip(frames, fps=int(1 / frame_interval))
    clip.write_videofile(output_path, codec="libx264", bitrate="12000k")


def main_runner():
    pr = cProfile.Profile()
    pr.enable()
    create_torus_animation(
        texture_path="space.jpg",
        output_path="torus_spacetime_animation.mp4",
        base_frames=600,
        speed=1.8,
        frame_interval=0.02,
        toroidal_cycles=2,
        view_elev=40,
    )
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumtime").print_stats(25)
    print(s.getvalue())


if __name__ == "__main__":
    main_runner()
