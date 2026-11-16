import type { LandmarkPoint } from '@/types/api';

/**
 * MediaPipe Hand landmark connections (21 landmarks, 0-indexed).
 * Standard skeleton connectivity.
 */
const HAND_CONNECTIONS: [number, number][] = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle
  [0, 9], [9, 10], [10, 11], [11, 12],
  // Ring
  [0, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [0, 17], [17, 18], [18, 19], [19, 20],
  // Palm
  [5, 9], [9, 13], [13, 17],
];

export interface RenderArea {
  /** X offset of the actual video content inside the canvas element (letterbox) */
  offsetX: number;
  /** Y offset of the actual video content inside the canvas element (letterbox) */
  offsetY: number;
  /** Width of the actual video content */
  contentWidth: number;
  /** Height of the actual video content */
  contentHeight: number;
}

/**
 * Calculates the actual video content area inside a container that uses object-fit: contain.
 */
export function getContainArea(
  containerWidth: number,
  containerHeight: number,
  videoWidth: number,
  videoHeight: number,
): RenderArea {
  if (!videoWidth || !videoHeight) {
    return { offsetX: 0, offsetY: 0, contentWidth: containerWidth, contentHeight: containerHeight };
  }
  const containerAspect = containerWidth / containerHeight;
  const videoAspect = videoWidth / videoHeight;

  let contentWidth: number;
  let contentHeight: number;

  if (containerAspect > videoAspect) {
    contentHeight = containerHeight;
    contentWidth = contentHeight * videoAspect;
  } else {
    contentWidth = containerWidth;
    contentHeight = contentWidth / videoAspect;
  }

  return {
    offsetX: (containerWidth - contentWidth) / 2,
    offsetY: (containerHeight - contentHeight) / 2,
    contentWidth,
    contentHeight,
  };
}

/**
 * Renders all 21 hand landmarks and connections onto a canvas.
 *
 * MIRRORING NOTE:
 * - The CSS video element is mirrored (scaleX(-1)) for natural display.
 * - The backend ALSO mirrors the frame before landmark detection.
 * - Therefore landmark x-coordinates are already aligned with the mirrored video.
 * - Do NOT apply an additional horizontal flip here.
 *
 * @param ctx - Canvas 2D rendering context
 * @param landmarks - Array of 21 normalized [0,1] x/y/z landmark points
 * @param area - The actual video content area (accounting for letterboxing)
 * @param dpr - Device pixel ratio for crisp rendering
 */
export function renderLandmarks(
  ctx: CanvasRenderingContext2D,
  landmarks: LandmarkPoint[],
  area: RenderArea,
  dpr = 1,
): void {
  if (!landmarks || landmarks.length !== 21) return;

  const { offsetX, offsetY, contentWidth, contentHeight } = area;

  function lmToPixel(lm: LandmarkPoint): [number, number] {
    return [
      (offsetX + lm.x * contentWidth) * dpr,
      (offsetY + lm.y * contentHeight) * dpr,
    ];
  }

  ctx.save();
  ctx.lineWidth = 2 * dpr;

  // Draw connections
  ctx.strokeStyle = 'rgba(139, 92, 246, 0.85)'; // violet-500
  ctx.shadowColor = 'rgba(139, 92, 246, 0.4)';
  ctx.shadowBlur = 4 * dpr;
  for (const [a, b] of HAND_CONNECTIONS) {
    const [ax, ay] = lmToPixel(landmarks[a]);
    const [bx, by] = lmToPixel(landmarks[b]);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  }

  // Draw landmark nodes
  ctx.shadowColor = 'rgba(34, 211, 238, 0.6)'; // cyan-400 glow
  ctx.shadowBlur = 6 * dpr;
  for (let i = 0; i < 21; i++) {
    const [px, py] = lmToPixel(landmarks[i]);
    const isKeyPoint = [0, 4, 8, 12, 16, 20].includes(i); // wrist + fingertips

    ctx.beginPath();
    ctx.arc(px, py, (isKeyPoint ? 5 : 3) * dpr, 0, Math.PI * 2);
    ctx.fillStyle = isKeyPoint ? '#22d3ee' : '#a78bfa'; // cyan-400 or violet-400
    ctx.fill();
  }

  ctx.restore();
}

/**
 * Clears the canvas completely.
 */
export function clearCanvas(canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D): void {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}
