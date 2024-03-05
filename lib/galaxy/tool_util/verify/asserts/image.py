import io
from typing import (
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
)

from ._util import _assert_number

try:
    import numpy
except ImportError:
    pass
try:
    from PIL import Image
except ImportError:
    pass

if TYPE_CHECKING:
    import numpy.typing


def _assert_float(
    actual: float,
    label: str,
    tolerance: Union[float, str],
    expected: Optional[Union[float, str]] = None,
    range_min: Optional[Union[float, str]] = None,
    range_max: Optional[Union[float, str]] = None,
) -> None:

    # Perform `tolerance` based check.
    if expected is not None:
        assert abs(actual - float(expected)) <= float(
            tolerance
        ), f"Wrong {label}: {actual} (expected {expected} ±{tolerance})"

    # Perform `range_min` based check.
    if range_min is not None:
        assert actual >= float(range_min), f"Wrong {label}: {actual} (must be {range_min} or larger)"

    # Perform `range_max` based check.
    if range_max is not None:
        assert actual <= float(range_max), f"Wrong {label}: {actual} (must be {range_max} or smaller)"


def assert_has_image_width(
    output_bytes: bytes,
    value: Optional[Union[int, str]] = None,
    delta: Union[int, str] = 0,
    min: Optional[Union[int, str]] = None,
    max: Optional[Union[int, str]] = None,
    negate: Union[bool, str] = False,
) -> None:
    """
    Asserts the specified output is an image and has a width of the specified value.
    """
    buf = io.BytesIO(output_bytes)
    with Image.open(buf) as im:
        _assert_number(
            im.size[0],
            value,
            delta,
            min,
            max,
            negate,
            "{expected} width {n}+-{delta}",
            "{expected} width to be in [{min}:{max}]",
        )


def assert_has_image_height(
    output_bytes: bytes,
    value: Optional[Union[int, str]] = None,
    delta: Union[int, str] = 0,
    min: Optional[Union[int, str]] = None,
    max: Optional[Union[int, str]] = None,
    negate: Union[bool, str] = False,
) -> None:
    """
    Asserts the specified output is an image and has a height of the specified value.
    """
    buf = io.BytesIO(output_bytes)
    with Image.open(buf) as im:
        _assert_number(
            im.size[1],
            value,
            delta,
            min,
            max,
            negate,
            "{expected} width {n}+-{delta}",
            "{expected} width to be in [{min}:{max}]",
        )


def assert_has_image_channels(
    output_bytes: bytes,
    value: Optional[Union[int, str]] = None,
    delta: Union[int, str] = 0,
    min: Optional[Union[int, str]] = None,
    max: Optional[Union[int, str]] = None,
    negate: Union[bool, str] = False,
) -> None:
    """
    Asserts the specified output is an image and has the specified number of channels.
    """
    buf = io.BytesIO(output_bytes)
    with Image.open(buf) as im:
        _assert_number(
            len(im.getbands()),
            value,
            delta,
            min,
            max,
            negate,
            "{expected} width {n}+-{delta}",
            "{expected} width to be in [{min}:{max}]",
        )


def _compute_center_of_mass(im_arr: "numpy.typing.NDArray") -> Tuple[float, float]:
    while im_arr.ndim > 2:
        im_arr = im_arr.sum(axis=2)
    im_arr = numpy.abs(im_arr)
    if im_arr.sum() == 0:
        return (numpy.nan, numpy.nan)
    im_arr = im_arr / im_arr.sum()
    yy, xx = numpy.indices(im_arr.shape)
    return (im_arr * xx).sum(), (im_arr * yy).sum()


def _get_image(
    output_bytes: bytes,
    channel: Optional[Union[int, str]] = None,
) -> "numpy.typing.NDArray":
    """
    Returns the output image or a specific channel.
    """

    buf = io.BytesIO(output_bytes)
    with Image.open(buf) as im:
        im_arr = numpy.array(im)

    # Select the specified channel (if any).
    if channel is not None:
        im_arr = im_arr[:, :, int(channel)]

    # Return the image
    return im_arr


def assert_has_image_mean_intensity(
    output_bytes: bytes,
    channel: Optional[Union[int, str]] = None,
    value: Optional[Union[float, str]] = None,
    delta: Union[float, str] = 0.01,
    min: Optional[Union[float, str]] = None,
    max: Optional[Union[float, str]] = None,
) -> None:
    """
    Asserts the specified output is an image and has the specified mean intensity value.
    """

    im_arr = _get_image(output_bytes, channel)
    _assert_float(
        actual=im_arr.mean(),
        label="mean intensity",
        tolerance=delta,
        expected=value,
        range_min=min,
        range_max=max,
    )


def assert_has_image_center_of_mass(
    output_bytes: bytes,
    channel: Optional[Union[int, str]] = None,
    point: Optional[Union[Tuple[float, float], str]] = None,
    delta: Union[float, str] = 0.01,
) -> None:
    """
    Asserts the specified output is an image and has the specified center of mass.
    """

    im_arr = _get_image(output_bytes, channel)
    if point is not None:
        if isinstance(point, str):
            point_parts = [c.strip() for c in point.split(",")]
            assert len(point_parts) == 2
            point = (float(point_parts[0]), float(point_parts[1]))
        assert len(point) == 2, "point must have two components"
        actual_center_of_mass = _compute_center_of_mass(im_arr)
        distance = numpy.linalg.norm(numpy.subtract(point, actual_center_of_mass))
        assert distance <= float(
            delta
        ), f"Wrong center of mass: {actual_center_of_mass} (expected {point}, distance: {distance}, delta: {delta})"


def assert_image_has_labels(
    output_bytes: bytes,
    number_of_objects: Optional[Union[int, str]] = None,
    mean_object_size: Optional[Union[float, str]] = None,
    mean_object_size_min: Optional[Union[float, str]] = None,
    mean_object_size_max: Optional[Union[float, str]] = None,
    exclude_labels: Optional[Union[str, List[int]]] = None,
    eps: Union[float, str] = 0.01,
) -> None:
    """
    Assert the image output has specific label content.
    """
    buf = io.BytesIO(output_bytes)
    with Image.open(buf) as im:
        im_arr = numpy.array(im)

    # Determine labels present in the image.
    labels = numpy.unique(im_arr)

    # Apply filtering due to `exclude_labels`.
    if exclude_labels is None:
        exclude_labels = list()
    if isinstance(exclude_labels, str):

        def cast_label(label):
            if numpy.issubdtype(im_arr.dtype, numpy.integer):
                return int(label)
            if numpy.issubdtype(im_arr.dtype, float):
                return float(label)
            raise AssertionError(f'Unsupported image label type: "{im_arr.dtype}"')

        exclude_labels = [cast_label(label) for label in exclude_labels.split(",") if len(label) > 0]
    labels = [label for label in labels if label not in exclude_labels]

    # Perform `number_of_objects` assertion.
    if number_of_objects is not None:
        actual_number_of_objects = len(labels)
        expected_number_of_objects = int(number_of_objects)
        assert (
            actual_number_of_objects == expected_number_of_objects
        ), f"Wrong number of objects: {actual_number_of_objects} (expected {expected_number_of_objects})"

    # Perform `mean_object_size` assertion.
    actual_mean_object_size = sum((im_arr == label).sum() for label in labels) / len(labels)
    _assert_float(
        actual=actual_mean_object_size,
        label="mean object size",
        tolerance=eps,
        expected=mean_object_size,
        range_min=mean_object_size_min,
        range_max=mean_object_size_max,
    )
