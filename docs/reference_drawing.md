# Reference drawing: physical ends A and B (versioned)

Version: `refdraw-1` · Applies to the geometry contract in `src/pvs/geometry.py` (PDF p.3).

## Convention

* Coordinates: rectified, unmirrored source image; origin top-left, +x right, +y down; clockwise positive.
* **Heading** = direction from physical end **A** to physical end **B**: `atan2(yB − yA, xB − xA)` wrapped to [0°, 360°).
* **Axis** = undirected long line, [0°, 180°).
* In the **canonical upright front view** (front logo reads upright) A is on the **left**, B on the **right**,
  so the heading of the canonical front reference is 0°.
* Turning the pack over about its long axis keeps A on the left in the **canonical back view**
  (A/B are physical ends, not image sides). Back references therefore also carry A = left, B = right.
* Turning end-over-end (rotation about the short axis) swaps the image positions of A and B; the
  heading changes by 180°.

## Fixture product "Zephyr" {#fixture-zephyr}

| End | Physical feature | Reference pixel (420×120 artwork) |
|---|---|---|
| A | solid dark notch band | (0, 60) |
| B | white zig-zag crimp | (420, 60) |

Variant-specific cue regions (reference pixels): front stripe `[40,8,380,26]`, front badge `[296,90,372,114]`,
back stripe `[40,8,380,16]`.

## Cadbury 5 Star — to be completed at enrolment (status: **not approved**)

The owner must approve, on a photographed physical pack of each enrolled revision:

1. which crimped end is **A** (proposal: the end on the left when the "5Star" wordmark reads upright on the
   front; on the observed retail wrapper this is the end nearest the Cadbury roundel),
2. the reference images (front, back; 8–16 clean views across sides, p.5) and their A/B pixel coordinates,
3. variant-specific cue regions (quantity text, flavour marks) that distinguish neighbouring SKUs,
4. the agreed endpoint-chord rule for moderately bent bars (default: chord between the two crimp centres;
   reject when mask fit residual > `axis_max_residual`).

Record the approval in the catalogue entry (`physical_ends.definition` → this file + revision).
