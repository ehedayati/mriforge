# File created by: Eisa Hedayati
# Date: 1/3/2024
# Description: This file is developed at CMRR
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import HBox, VBox, interactive, widgets
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from scipy.ndimage import zoom

from mrif.utilities.numerical import resize_matrix


def get_vmin_vmax(images, vmin=None, vmax=None, ignore_nan=True):
    """
    Determine vmin/vmax for a list of images if not explicitly provided.
    """
    if vmin is not None and vmax is not None:
        return vmin, vmax

    imgs = np.asarray(images, dtype=float)

    if ignore_nan:
        data_min = np.nanmin(imgs)
        data_max = np.nanmax(imgs)
    else:
        data_min = imgs.min()
        data_max = imgs.max()

    vmin = data_min if vmin is None else vmin
    vmax = data_max if vmax is None else vmax

    return vmin, vmax

def _get_valid_cmap(color_palette: str):
    available = plt.colormaps()

    if color_palette in available:
        return color_palette

    # Fallback
    print(
        f"[side_by_side_view] Warning: invalid colormap '{color_palette}'. "
        "Falling back to 'gray'."
    )
    return 'gray'


def side_by_side_view(*images: np.ndarray,
                      color_palette: str = 'gray',
                      dpi: int = 100,
                      titles: Optional[List[str]] = None,
                      show_colorbar: bool = True, force_square: bool = False, vmin=None, vmax=None):
    """
    Display multiple images side by side with an optional shared colorbar.
    """
    images_list = list(images)
    n_images = len(images_list)

    vmin, vmax = get_vmin_vmax(images_list[0], vmin, vmax)

    if n_images == 0:
        raise ValueError("At least one image must be provided.")

    if titles is None or len(titles) != n_images:
        titles = [f"image {i+1}" for i in range(n_images)]

    fig, axs = plt.subplots(
        1, n_images,
        figsize=(5 * n_images, 5),
        dpi=dpi,
        constrained_layout=True
    )

    if n_images == 1:
        axs = [axs]

    cmap = _get_valid_cmap(color_palette)

    im = None  # keep handle for colorbar

    for idx, (img, title) in enumerate(zip(images_list, titles, strict=True)):
        if force_square:
            if img.shape[0] != img.shape[1]:
                img = resize_matrix(img)

        im = axs[idx].imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[idx].axis('off')
        axs[idx].set_title(title)

    if show_colorbar and im is not None:
        cbar = fig.colorbar(
            im,
            ax=axs,
            fraction=0.025,
            pad=0.02
        )
        cbar.ax.tick_params(labelsize=8)

    # plt.tight_layout()
    plt.show()


def visualize_segmentation_slice(grayscale_image, segmentation_matrix, slice_number, dimension='axial', target_shape=(600, 600)):
    """
    Visualize a specific slice of the 3D segmentation matrix on top of the corresponding grayscale image slice.

    Args:
        grayscale_image (numpy.ndarray): 3D grayscale image matrix.
        segmentation_matrix (numpy.ndarray): 3D segmentation matrix.
        slice_number (int): The specific slice to visualize.
        dimension (str): The dimension along which to slice ('axial', 'coronal', 'sagittal').
        target_shape (tuple): The target shape for resizing the slices.
    """

    def extract_slice(matrix, slice_num, dim):
        """
        Extract a specific slice from a 3D matrix based on the specified dimension.

        Args:
            matrix (numpy.ndarray): The input 3D matrix.
            slice_num (int): The specific slice to extract.
            dim (str): The dimension along which to slice ('axial', 'coronal', 'sagittal').

        Returns:
            numpy.ndarray: The extracted 2D slice.
        """
        if dim == 'coronal':
            return matrix[:, slice_num, :]
        elif dim == 'axial':
            return matrix[slice_num, :, :]
        else:  # 'sagittal' or default
            return matrix[:, :, slice_num]

    # Define a color map for 10 distinct colors
    colors = np.array([
        [0, 0, 0],       # Color for 0
        [255, 0, 0],     # Color for 1
        [0, 255, 0],     # Color for 2
        [0, 0, 255],     # Color for 3
        [255, 255, 0],   # Color for 4
        [255, 0, 255],   # Color for 5
        [0, 255, 255],   # Color for 6
        [128, 0, 0],     # Color for 7
        [0, 128, 0],     # Color for 8
        [0, 0, 128]      # Color for 9
    ])

    # Extract and resize the specific slices
    grayscale_slice = extract_slice(grayscale_image, slice_number, dimension)
    segmentation_slice = extract_slice(segmentation_matrix, slice_number, dimension)

    grayscale_slice_resized = resize_matrix(grayscale_slice, target_shape)
    segmentation_slice_resized = resize_matrix(segmentation_slice, target_shape)

    # Apply color map
    segmentation_colored = np.zeros((*segmentation_slice_resized.shape, 3), dtype=np.uint8)
    for label in range(10):
        segmentation_colored[segmentation_slice_resized == label] = colors[label]
    plt.figure(dpi=100)
    # Overlay the segmentation on the grayscale image
    plt.imshow(grayscale_slice_resized, cmap='gray')
    plt.imshow(segmentation_colored, alpha=0.5)  # Adjust alpha for transparency

    # Display the result
    plt.axis('off')
    plt.show()



# Fixed palette for 3D segmentation labels: 0 is the transparent background.
# The label-to-color mapping is independent of the current slice's values.
_SEGMENTATION_COLORS = [
    (0, 0, 0, 0),
    (1, 0, 0, 1),          # 1: red
    (0, 1, 0, 1),          # 2: green
    (0, 0, 1, 1),          # 3: blue
    (1, 1, 0, 1),          # 4: yellow
    (1, 0, 1, 1),          # 5: magenta
    (0, 1, 1, 1),          # 6: cyan
    (1, 0.5, 0, 1),        # 7: orange
    (0.5, 0, 1, 1),        # 8: purple
    (0, 1, 0.5, 1),        # 9: teal
    (1, 0.5, 0.5, 1),      # 10: pink
    (0.6, 0.4, 0.2, 1),    # 11: brown
    (0, 0, 0.5, 1),        # 12: navy
    (0.5, 0.5, 0, 1),      # 13: olive
    (0.5, 0.5, 0.5, 1),    # 14: gray
    (0.5, 0.5, 1, 1),      # 15: light blue
    (1, 0.84, 0, 1),       # 16: gold
    (1, 0.6, 0.6, 1),      # 17: salmon
    (0.93, 0.51, 0.93, 1),# 18: violet
    (0.75, 1, 0, 1),       # 19: lime
]
_SEGMENTATION_CMAP = ListedColormap(_SEGMENTATION_COLORS)
_SEGMENTATION_NORM = BoundaryNorm(
    np.arange(-0.5, len(_SEGMENTATION_COLORS) + 0.5, 1),
    _SEGMENTATION_CMAP.N,
)


def _add_segmentation_legend(fig, ax, segmentation, show_legend):
    """Add a volume-wide, stable legend for labels represented by the palette."""
    if not show_legend or segmentation is None:
        return

    label_values = np.unique(segmentation)
    labels = [
        int(value)
        for value in label_values
        if np.isfinite(value)
        and value == int(value)
        and 0 < value < len(_SEGMENTATION_COLORS)
    ]
    if not labels:
        return

    # A compact, horizontal legend beneath the image. Wrap after seven labels
    # to keep the legend within the normal figure width (up to three rows for
    # the 19 labels supported by this palette).
    ncols = min(7, len(labels))
    nrows = (len(labels) + ncols - 1) // ncols
    bottom_margin = 0.02 + 0.03 * nrows
    ax.set_position([0, bottom_margin, 1, 1 - bottom_margin])

    # Matplotlib fills legend columns top-to-bottom. Arrange handles so the
    # labels read from left to right in increasing order on each row.
    ordered_labels = [
        labels[row * ncols + col]
        for col in range(ncols)
        for row in range(nrows)
        if row * ncols + col < len(labels)
    ]
    handles = [
        Patch(facecolor=_SEGMENTATION_COLORS[label][:3], label=f"Label {label}")
        for label in ordered_labels
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        bbox_transform=fig.transFigure,
        ncol=ncols,
        frameon=False,
        fontsize=8,
        borderaxespad=0,
        borderpad=0,
        labelspacing=0.2,
        columnspacing=0.7,
        handlelength=1.0,
        handletextpad=0.35,
    )


def _resize_3d_slice(mri_slice, seg_slice, target_shape):
    """Resize MRI continuously but segmentation categorically."""
    if target_shape is not None:
        mri_slice = resize_matrix(mri_slice, target_shape)
        if seg_slice is not None:
            # Nearest-neighbor prevents creation of spurious label values.
            factors = (
                target_shape[0] / seg_slice.shape[0],
                target_shape[1] / seg_slice.shape[1],
            )
            seg_slice = zoom(seg_slice, factors, order=0)
    return mri_slice, seg_slice


def _display_3d_segmentation(ax, seg_slice, alpha):
    return ax.imshow(
        seg_slice,
        cmap=_SEGMENTATION_CMAP,
        norm=_SEGMENTATION_NORM,
        alpha=alpha,
        interpolation="nearest",
    )


def plot_3D_mri(
    mri_image, slice_number=None, direction='sagittal', segmentation=None,
    alpha=0.5, dpi=150, target_shape=None, cmap='gray', vmax=None, vmin=None,
    show_legend=False,
):
    """View 3D MRI slices, optionally overlaying a labeled segmentation.

    Parameters
    ----------
    show_legend : bool, optional
        Show colors for the nonzero labels present anywhere in `segmentation`.
        False by default. The original palette supports label IDs 1 through 19.
    """
    backend = matplotlib.get_backend().lower()
    interactive_backend = ("ipympl" in backend) or ("qt" in backend)

    if not interactive_backend:
        print("Note: interactive backend not detected; falling back to static redraw mode. Use %matplotlib widget if available")
        return plot_3D_mri_inline(
            mri_image, slice_number, direction, segmentation, alpha, dpi,
            target_shape, cmap, vmax, vmin, show_legend=show_legend,
        )
    return plot_3D_mri_interactive(
        mri_image, slice_number, direction, segmentation, alpha, dpi,
        target_shape, cmap, vmax, vmin, show_legend=show_legend,
    )


def plot_3D_mri_interactive(
    mri_image, slice_number=None, direction='sagittal', segmentation=None,
    alpha=0.5, dpi=150, target_shape=None, m_cmap='gray', vmax=None, vmin=None,
    show_legend=False,
):
    """Display a reusable Matplotlib figure with optional segmentation legend."""
    if len(mri_image.shape) == 4:
        mri_image = mri_image[..., 0]
    if vmax is None:
        vmax = np.nanmax(mri_image)
    if vmin is None:
        vmin = np.nanmin(mri_image)

    if direction == 'axial':
        max_slices = mri_image.shape[0]
        slice_func = lambda i: (mri_image[i, :, :], segmentation[i, :, :] if segmentation is not None else None)
    elif direction == 'coronal':
        max_slices = mri_image.shape[1]
        slice_func = lambda i: (mri_image[:, i, :], segmentation[:, i, :] if segmentation is not None else None)
    elif direction == 'sagittal':
        max_slices = mri_image.shape[2]
        slice_func = lambda i: (mri_image[:, :, i], segmentation[:, :, i] if segmentation is not None else None)
    else:
        raise ValueError("Direction must be one of 'axial', 'coronal', or 'sagittal'.")

    def get_slice(slice_index):
        return _resize_3d_slice(*slice_func(slice_index), target_shape)

    def create_figure():
        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.set_position([0, 0, 1, 1])
        _add_segmentation_legend(fig, ax, segmentation, show_legend)
        return fig, ax

    if slice_number is not None:
        mri_slice, seg_slice = get_slice(slice_number)
        fig, ax = create_figure()
        ax.imshow(mri_slice, cmap=m_cmap, vmin=vmin, vmax=vmax)
        if seg_slice is not None:
            _display_3d_segmentation(ax, seg_slice, alpha)
        plt.show()
        return

    # Reuse a single figure and its image artists as the slider changes.
    slider = widgets.IntSlider(min=0, max=max_slices - 1, step=1, value=max_slices // 2)

    def next_slice(_):
        slider.value = min(slider.value + 1, max_slices - 1)

    def prev_slice(_):
        slider.value = max(slider.value - 1, 0)

    button_next = widgets.Button(description="Next")
    button_prev = widgets.Button(description="Previous")
    button_next.on_click(next_slice)
    button_prev.on_click(prev_slice)
    display(HBox([button_prev, button_next, slider]))

    init_mri, init_seg = get_slice(slider.value)
    fig, ax = create_figure()
    mri_im = ax.imshow(init_mri, cmap=m_cmap, vmin=vmin, vmax=vmax)
    seg_im = None
    if init_seg is not None:
        seg_im = _display_3d_segmentation(ax, init_seg, alpha)
    plt.show()

    def update_slice(slice_index):
        nonlocal seg_im
        mri_slice, seg_slice = get_slice(slice_index)
        mri_im.set_data(mri_slice)
        if segmentation is not None:
            if seg_im is None and seg_slice is not None:
                seg_im = _display_3d_segmentation(ax, seg_slice, alpha)
            elif seg_im is not None:
                if seg_slice is None:
                    seg_im.set_visible(False)
                else:
                    seg_im.set_visible(True)
                    seg_im.set_data(seg_slice)
        fig.canvas.draw_idle()

    def _on_slider_change(change):
        if change.get("name") == "value":
            update_slice(change["new"])

    slider.observe(_on_slider_change, names="value")


def plot_3D_mri_inline(
    mri_image, slice_number=None, direction='sagittal', segmentation=None,
    alpha=0.5, dpi=150, target_shape=None, m_cmap='gray', vmax=None, vmin=None,
    show_legend=False,
):
    """Render 3D MRI slices with optional segmentation and optional legend.

    If `slice_number` is omitted, an ipywidgets slider redraws each slice.
    `show_legend=True` displays nonzero segmentation labels across the volume.
    """
    if len(mri_image.shape) == 4:
        mri_image = mri_image[..., 0]
    if vmax is None:
        vmax = np.nanmax(mri_image)
    if vmin is None:
        vmin = np.nanmin(mri_image)

    if direction == 'axial':
        max_slices = mri_image.shape[0]
        slice_func = lambda i: (mri_image[i, :, :], segmentation[i, :, :] if segmentation is not None else None)
    elif direction == 'coronal':
        max_slices = mri_image.shape[1]
        slice_func = lambda i: (mri_image[:, i, :], segmentation[:, i, :] if segmentation is not None else None)
    elif direction == 'sagittal':
        max_slices = mri_image.shape[2]
        slice_func = lambda i: (mri_image[:, :, i], segmentation[:, :, i] if segmentation is not None else None)
    else:
        raise ValueError("Direction must be one of 'axial', 'coronal', or 'sagittal'.")

    def plot_slice(slice_index):
        mri_slice, seg_slice = _resize_3d_slice(*slice_func(slice_index), target_shape)
        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
        ax.imshow(mri_slice, cmap=m_cmap, vmin=vmin, vmax=vmax)
        if seg_slice is not None:
            _display_3d_segmentation(ax, seg_slice, alpha)
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.set_position([0, 0, 1, 1])
        _add_segmentation_legend(fig, ax, segmentation, show_legend)
        plt.show()

    if slice_number is not None:
        plot_slice(slice_number)
    else:
        slider = widgets.IntSlider(min=0, max=max_slices - 1, step=1, value=max_slices // 2)

        def next_slice(_):
            slider.value = min(slider.value + 1, max_slices - 1)

        def prev_slice(_):
            slider.value = max(slider.value - 1, 0)

        button_next = widgets.Button(description="Next")
        button_prev = widgets.Button(description="Previous")
        button_next.on_click(next_slice)
        button_prev.on_click(prev_slice)

        controls = HBox([button_prev, button_next])
        interactive_plot = interactive(plot_slice, slice_index=slider)
        display(VBox([controls, interactive_plot]))
