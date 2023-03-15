from typing import Optional, Union, NamedTuple, Callable, Sequence
from functools import partial
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import numpy as np
from .metrics import compute_precisions


class ContactAndAttentionArtists(NamedTuple):
    image: mpl.image.AxesImage
    contacts: mpl.lines.Line2D
    false_positives: mpl.lines.Line2D
    true_positives: mpl.lines.Line2D
    title: Optional[mpl.text.Text] = None


def plot_attentions(
    attentions: Union[torch.Tensor, np.ndarray],
    ax: Optional[mpl.axes.Axes] = None,
    img: Optional[mpl.image.AxesImage] = None,
    cmap: str = "Blues",
) -> mpl.image.AxesImage:

    if isinstance(attentions, torch.Tensor):
        attentions = attentions.detach().cpu().numpy()
    if ax is None:
        ax = plt.gca()

    if img is not None:
        img.set_data(attentions)
    else:
        img = ax.imshow(attentions, cmap=cmap)
    return img


def plot_contacts_and_attentions(
    predictions: Union[torch.Tensor, np.ndarray],
    contacts: Union[torch.Tensor, np.ndarray],
    ax: Optional[mpl.axes.Axes] = None,
    artists: Optional[ContactAndAttentionArtists] = None,
    cmap: str = "Blues",
    ms: float = 1,
    title: Union[bool, str, Callable[[float], str]] = True,
    animated: bool = False,
) -> ContactAndAttentionArtists:

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(contacts, torch.Tensor):
        contacts = contacts.detach().cpu().numpy()
    if ax is None:
        ax = plt.gca()

    seqlen = contacts.shape[0]
    relative_distance = np.add.outer(-np.arange(seqlen), np.arange(seqlen))
    bottom_mask = relative_distance < 0
    masked_image = np.ma.masked_where(bottom_mask, predictions)
    invalid_mask = np.abs(np.add.outer(np.arange(seqlen), -np.arange(seqlen))) < 6
    predictions = predictions.copy()
    predictions[invalid_mask] = float("-inf")

    topl_val = np.sort(predictions.reshape(-1))[-seqlen]
    pred_contacts = predictions >= topl_val
    true_positives = contacts & pred_contacts & ~bottom_mask
    false_positives = ~contacts & pred_contacts & ~bottom_mask
    other_contacts = contacts & ~pred_contacts & ~bottom_mask

    if isinstance(title, str):
        title_text: Optional[str] = title
    elif title:
        long_range_pl = compute_precisions(predictions, contacts, minsep=24)[
            "P@L"
        ].item()
        if callable(title):
            title_text = title(long_range_pl)
        else:
            title_text = f"Long Range P@L: {100 * long_range_pl:0.1f}"
    else:
        title_text = None

    if artists is not None:
        artists.image.set_data(masked_image)
        artists.contacts.set_data(*np.where(other_contacts))
        artists.false_positives.set_data(*np.where(false_positives))
        artists.true_positives.set_data(*np.where(true_positives))
        if artists.title is not None and title_text is not None:
            artists.title.set_data(title_text)
    else:
        img = ax.imshow(masked_image, cmap=cmap, animated=animated)
        oc = ax.plot(*np.where(other_contacts), "o", c="grey", ms=ms)[0]
        fn = ax.plot(*np.where(false_positives), "o", c="r", ms=ms)[0]
        tp = ax.plot(*np.where(true_positives), "o", c="b", ms=ms)[0]
        ti = ax.set_title(title_text) if title_text is not None else None
        artists = ContactAndAttentionArtists(img, oc, fn, tp, ti)

        ax.axis("square")
        ax.set_xlim([0, seqlen])
        ax.set_ylim([0, seqlen])
    return artists


def animate_contacts_and_attentions(
    predictions: Sequence[Union[torch.Tensor, np.ndarray]],
    contacts: Union[torch.Tensor, np.ndarray],
    fig: Optional[mpl.figure.Figure] = None,
    ax: Optional[mpl.axes.Axes] = None,
    artists: Optional[ContactAndAttentionArtists] = None,
    cmap: str = "Blues",
    ms: float = 1,
    title: Union[bool, str, Callable[[int, float], str]] = True,
    interval: int = 500,
    repeat_delay: int = 1000,
    blit: bool = True,
) -> mpl.animation.Animation:
    if fig is None:
        fig = plt.gcf()

    if ax is None:
        ax = plt.gca()

    initial_title = partial(title, 0) if callable(title) else title
    artists = plot_contacts_and_attentions(
        predictions[0],
        contacts,
        ax,
        cmap=cmap,
        ms=ms,
        title=initial_title,
        animated=True,
    )

    def update(i):
        iter_title = partial(title, i) if callable(title) else title
        return plot_contacts_and_attentions(
            predictions[i],
            contacts,
            ax,
            artists=artists,
            cmap=cmap,
            ms=ms,
            title=iter_title,
            animated=True,
        )

    ani = mpl.animation.FuncAnimation(
        fig,
        update,
        len(predictions),
        interval=interval,
        blit=blit,
        repeat_delay=repeat_delay,
    )
    return ani
