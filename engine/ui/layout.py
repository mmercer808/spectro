"""
Layout Engine

Flexbox-inspired layout system.

Implements a subset of CSS Flexbox:
- direction (row/column)
- justify-content (main axis distribution)
- align-items (cross axis alignment)
- gap
- flex-grow/flex-shrink on children

Not implemented (to keep it simple):
- flex-wrap
- align-content
- order
- flex-basis (uses width/height instead)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, TYPE_CHECKING
from enum import Enum, auto

if TYPE_CHECKING:
    from engine.ui.widget import Widget
    from engine.ui.style import Style


# =============================================================================
# Enums
# =============================================================================

class LayoutDirection(Enum):
    ROW = auto()      # Horizontal, left to right
    COLUMN = auto()   # Vertical, top to bottom


class Justify(Enum):
    """Main axis distribution."""
    START = auto()
    END = auto()
    CENTER = auto()
    SPACE_BETWEEN = auto()
    SPACE_AROUND = auto()
    SPACE_EVENLY = auto()


class Align(Enum):
    """Cross axis alignment."""
    START = auto()
    END = auto()
    CENTER = auto()
    STRETCH = auto()


# =============================================================================
# Rect
# =============================================================================

@dataclass
class Rect:
    """Rectangle with position and size."""
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px < (self.x + self.w) and self.y <= py < (self.y + self.h)

    def inset(self, top: float, right: float, bottom: float, left: float) -> Rect:
        """Return new rect inset by the given amounts."""
        return Rect(
            x=self.x + left,
            y=self.y + top,
            w=max(0, self.w - left - right),
            h=max(0, self.h - top - bottom),
        )

    def inset_by(self, insets) -> Rect:
        """Return new rect inset by EdgeInsets."""
        return self.inset(insets.top, insets.right, insets.bottom, insets.left)

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def center_x(self) -> float:
        return self.x + self.w / 2

    @property
    def center_y(self) -> float:
        return self.y + self.h / 2

    def copy(self) -> Rect:
        return Rect(self.x, self.y, self.w, self.h)


# =============================================================================
# Size Constraints
# =============================================================================

@dataclass
class Constraints:
    """
    Size constraints for layout.
    Similar to Flutter's BoxConstraints.
    """
    min_w: float = 0.0
    min_h: float = 0.0
    max_w: float = float('inf')
    max_h: float = float('inf')

    def constrain(self, w: float, h: float) -> Tuple[float, float]:
        """Constrain a size to these constraints."""
        return (
            max(self.min_w, min(self.max_w, w)),
            max(self.min_h, min(self.max_h, h)),
        )

    def constrain_width(self, w: float) -> float:
        return max(self.min_w, min(self.max_w, w))

    def constrain_height(self, h: float) -> float:
        return max(self.min_h, min(self.max_h, h))

    @staticmethod
    def tight(w: float, h: float) -> Constraints:
        """Exact size constraints."""
        return Constraints(w, h, w, h)

    @staticmethod
    def loose(max_w: float, max_h: float) -> Constraints:
        """Maximum size constraints with no minimum."""
        return Constraints(0, 0, max_w, max_h)

    def loosen(self) -> Constraints:
        """Remove minimum constraints."""
        return Constraints(0, 0, self.max_w, self.max_h)


# =============================================================================
# Flex Layout
# =============================================================================

@dataclass
class FlexLayout:
    """
    Flexbox-style layout configuration.

    Attach to a widget to control how its children are laid out.
    """
    direction: LayoutDirection = LayoutDirection.COLUMN
    justify: Justify = Justify.START
    align: Align = Align.STRETCH
    gap: float = 0.0

    def is_row(self) -> bool:
        return self.direction == LayoutDirection.ROW

    def is_column(self) -> bool:
        return self.direction == LayoutDirection.COLUMN


# =============================================================================
# Layout Algorithm
# =============================================================================

def layout_flex(
    children: List[Widget],
    available: Rect,
    flex: FlexLayout,
) -> List[Rect]:
    """
    Perform flexbox layout on children within available rect.

    Returns list of Rects, one per child.

    Algorithm:
    1. Measure all children with constraints
    2. Compute main axis sizes (respecting flex-grow/shrink)
    3. Position children along main axis (respecting justify)
    4. Position children along cross axis (respecting align)
    """
    if not children:
        return []

    is_row = flex.is_row()
    main_size = available.w if is_row else available.h
    cross_size = available.h if is_row else available.w

    # --- Step 1: Measure children ---
    # First pass: get preferred sizes
    child_main_sizes: List[float] = []
    child_cross_sizes: List[float] = []
    flex_grows: List[float] = []
    flex_shrinks: List[float] = []

    for child in children:
        style = child.style
        measured_w, measured_h = child.measure(
            Constraints.loose(available.w, available.h)
        )

        if is_row:
            child_main_sizes.append(measured_w)
            child_cross_sizes.append(measured_h)
        else:
            child_main_sizes.append(measured_h)
            child_cross_sizes.append(measured_w)

        flex_grows.append(style.flex_grow)
        flex_shrinks.append(style.flex_shrink)

    # --- Step 2: Distribute main axis space ---
    n = len(children)
    total_gap = flex.gap * (n - 1) if n > 1 else 0
    content_main = sum(child_main_sizes)
    available_main = main_size - total_gap

    # How much space to distribute?
    extra_space = available_main - content_main

    final_main_sizes = child_main_sizes.copy()

    if extra_space > 0:
        # Grow: distribute extra space according to flex_grow
        total_grow = sum(flex_grows)
        if total_grow > 0:
            for i, grow in enumerate(flex_grows):
                if grow > 0:
                    final_main_sizes[i] += extra_space * (grow / total_grow)
    elif extra_space < 0:
        # Shrink: reduce sizes according to flex_shrink
        shrink_space = -extra_space
        total_shrink = sum(
            flex_shrinks[i] * child_main_sizes[i]
            for i in range(n)
        )
        if total_shrink > 0:
            for i in range(n):
                shrink_factor = flex_shrinks[i] * child_main_sizes[i]
                reduction = shrink_space * (shrink_factor / total_shrink)
                final_main_sizes[i] = max(0, final_main_sizes[i] - reduction)

    # --- Step 3: Position along main axis ---
    total_used = sum(final_main_sizes) + total_gap
    remaining = available_main - sum(final_main_sizes)

    # Starting position based on justify
    if flex.justify == Justify.START:
        main_pos = 0.0
        spacing = 0.0
    elif flex.justify == Justify.END:
        main_pos = remaining
        spacing = 0.0
    elif flex.justify == Justify.CENTER:
        main_pos = remaining / 2
        spacing = 0.0
    elif flex.justify == Justify.SPACE_BETWEEN:
        main_pos = 0.0
        spacing = remaining / (n - 1) if n > 1 else 0
    elif flex.justify == Justify.SPACE_AROUND:
        spacing = remaining / n if n > 0 else 0
        main_pos = spacing / 2
    elif flex.justify == Justify.SPACE_EVENLY:
        spacing = remaining / (n + 1) if n > 0 else 0
        main_pos = spacing
    else:
        main_pos = 0.0
        spacing = 0.0

    # --- Step 4: Build final rects ---
    results: List[Rect] = []

    for i, child in enumerate(children):
        child_main = final_main_sizes[i]
        child_cross = child_cross_sizes[i]
        style = child.style

        # Cross axis alignment
        align = flex.align
        if style.align_self != "auto":
            align = {
                "start": Align.START,
                "end": Align.END,
                "center": Align.CENTER,
                "stretch": Align.STRETCH,
            }.get(style.align_self, align)

        if align == Align.START:
            cross_pos = 0.0
            final_cross = child_cross
        elif align == Align.END:
            cross_pos = cross_size - child_cross
            final_cross = child_cross
        elif align == Align.CENTER:
            cross_pos = (cross_size - child_cross) / 2
            final_cross = child_cross
        else:  # STRETCH
            cross_pos = 0.0
            final_cross = cross_size

        # Build rect
        if is_row:
            rect = Rect(
                x=available.x + main_pos,
                y=available.y + cross_pos,
                w=child_main,
                h=final_cross,
            )
        else:
            rect = Rect(
                x=available.x + cross_pos,
                y=available.y + main_pos,
                w=final_cross,
                h=child_main,
            )

        results.append(rect)
        main_pos += child_main + flex.gap + spacing

    return results


def measure_flex(
    children: List[Widget],
    flex: FlexLayout,
    constraints: Constraints,
) -> Tuple[float, float]:
    """
    Measure the size needed for a flex container.

    Returns (width, height) that fits all children.
    """
    if not children:
        return (0.0, 0.0)

    is_row = flex.is_row()
    n = len(children)
    total_gap = flex.gap * (n - 1) if n > 1 else 0

    main_total = 0.0
    cross_max = 0.0

    for child in children:
        w, h = child.measure(constraints.loosen())

        if is_row:
            main_total += w
            cross_max = max(cross_max, h)
        else:
            main_total += h
            cross_max = max(cross_max, w)

    main_total += total_gap

    if is_row:
        return constraints.constrain(main_total, cross_max)
    else:
        return constraints.constrain(cross_max, main_total)
