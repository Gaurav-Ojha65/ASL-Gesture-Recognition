/**
 * Logo — 36×36px teal rounded square with a white open-hand glyph.
 * Pure SVG, no external assets required.
 */
export function Logo({ size = 36 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 36 36"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      {/* Background square */}
      <rect width="36" height="36" rx="10" fill="#0F5C5C" />

      {/* Palm */}
      <rect x="13" y="21" width="10" height="8" rx="3" fill="white" />

      {/* Fingers — index, middle, ring, pinky */}
      <rect x="13" y="10" width="2.5" height="12" rx="1.25" fill="white" />
      <rect x="16.75" y="8" width="2.5" height="14" rx="1.25" fill="white" />
      <rect x="20.5" y="9" width="2.5" height="13" rx="1.25" fill="white" />
      <rect x="24.25" y="11" width="2.5" height="11" rx="1.25" fill="white" />

      {/* Thumb */}
      <rect x="9" y="16" width="5" height="2.5" rx="1.25" fill="white" />
    </svg>
  );
}
