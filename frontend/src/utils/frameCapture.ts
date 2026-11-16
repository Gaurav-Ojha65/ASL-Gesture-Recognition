/**
 * Captures a JPEG frame from a video element without applying CSS transforms.
 * Explicitly downscales high-resolution video feeds (e.g. 1280x720) to 640x480
 * before JPEG encoding to eliminate transmission latency and maintain 12+ FPS throughput.
 * The backend will apply its own horizontal mirror before landmark detection.
 *
 * @param video - HTMLVideoElement with an active stream
 * @param maxWidth - Maximum width constraint (default 640)
 * @param maxHeight - Maximum height constraint (default 480)
 * @param quality - JPEG quality 0–1 (default 0.75)
 * @returns ArrayBuffer of JPEG bytes, or null if capture fails
 */
export async function captureJpegFrame(
  video: HTMLVideoElement,
  maxWidth = 640,
  maxHeight = 480,
  quality = 0.75,
): Promise<ArrayBuffer | null> {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return null;

  // Scale down while preserving aspect ratio if needed
  let dw = vw;
  let dh = vh;
  if (vw > maxWidth || vh > maxHeight) {
    const scale = Math.min(maxWidth / vw, maxHeight / vh);
    dw = Math.round(vw * scale);
    dh = Math.round(vh * scale);
  }

  const canvas = document.createElement('canvas');
  canvas.width = dw;
  canvas.height = dh;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  // Draw video WITHOUT CSS mirror transform — backend mirrors internally
  ctx.drawImage(video, 0, 0, dw, dh);

  return new Promise<ArrayBuffer | null>((resolve) => {
    canvas.toBlob(
      (blob) => {
        if (!blob) { resolve(null); return; }
        blob.arrayBuffer().then(resolve).catch(() => resolve(null));
      },
      'image/jpeg',
      quality,
    );
  });
}
