import type { LandmarkPoint } from '@/types/api';

/**
 * MediaPipe Hand landmark connections (21 landmarks, 0-indexed).
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
  offsetX: number;
  offsetY: number;
  contentWidth: number;
  contentHeight: number;
}

/** Canvas colour tokens for 2D landmark overlay */
export const CANVAS_COLORS = {
  boneOutline: 'rgba(31, 42, 46, 0.25)',
  boneFill: 'rgba(255, 255, 255, 0.60)',
  jointStroke: '#FFFFFF',
  jointFill: '#B23E26',
} as const;

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
 * Renders all 21 hand landmarks onto the canvas.
 *
 * MIRRORING: CSS video is scaleX(-1). Backend mirrors frame before detection.
 * Landmark x-coords already align with the mirrored video — do NOT re-flip.
 *
 * Color scheme (warm light redesign):
 *   Bones: white at 60% opacity, thin dark outline
 *   Joints: #B23E26 (coral-text) fill, white stroke
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

  // ── Bones (connections) ───────────────────────────────────────────────────
  ctx.lineWidth = 2.5 * dpr;
  ctx.lineCap = 'round';

  // Thin dark outline pass for legibility on light backgrounds
  ctx.strokeStyle = CANVAS_COLORS.boneOutline;
  ctx.lineWidth = 4 * dpr;
  for (const [a, b] of HAND_CONNECTIONS) {
    const [ax, ay] = lmToPixel(landmarks[a]);
    const [bx, by] = lmToPixel(landmarks[b]);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  }

  // White fill pass at 60% opacity
  ctx.strokeStyle = CANVAS_COLORS.boneFill;
  ctx.lineWidth = 2.5 * dpr;
  for (const [a, b] of HAND_CONNECTIONS) {
    const [ax, ay] = lmToPixel(landmarks[a]);
    const [bx, by] = lmToPixel(landmarks[b]);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  }

  // ── Joints (landmark nodes) ───────────────────────────────────────────────
  for (let i = 0; i < 21; i++) {
    const [px, py] = lmToPixel(landmarks[i]);
    const isKeyPoint = [0, 4, 8, 12, 16, 20].includes(i); // wrist + fingertips
    const r = (isKeyPoint ? 5 : 3.5) * dpr;

    // White stroke border
    ctx.beginPath();
    ctx.arc(px, py, r + 1 * dpr, 0, Math.PI * 2);
    ctx.fillStyle = CANVAS_COLORS.jointStroke;
    ctx.fill();

    // Coral fill
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    ctx.fillStyle = CANVAS_COLORS.jointFill;
    ctx.fill();
  }

  ctx.restore();
}

export function clearCanvas(canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D): void {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}
