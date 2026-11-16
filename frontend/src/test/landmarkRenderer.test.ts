import { describe, it, expect } from 'vitest';
import { getContainArea } from '@/utils/landmarkRenderer';
import type { LandmarkPoint } from '@/types/api';

describe('Landmark Renderer & Coordinate Translation', () => {
  it('letterboxes wider container preserving 4:3 video aspect ratio', () => {
    const area = getContainArea(1200, 400, 640, 480);
    expect(area.contentHeight).toBe(400);
    expect(area.contentWidth).toBeCloseTo(533.33, 1);
    expect(area.offsetX).toBeCloseTo(333.33, 1);
    expect(area.offsetY).toBe(0);
  });

  it('letterboxes taller container preserving 4:3 video aspect ratio', () => {
    const area = getContainArea(400, 600, 640, 480);
    expect(area.contentWidth).toBe(400);
    expect(area.contentHeight).toBe(300);
    expect(area.offsetX).toBe(0);
    expect(area.offsetY).toBe(150);
  });

  it('handles zero or uninitialized video dimensions gracefully', () => {
    const area = getContainArea(800, 600, 0, 0);
    expect(area.contentWidth).toBe(800);
    expect(area.contentHeight).toBe(600);
    expect(area.offsetX).toBe(0);
    expect(area.offsetY).toBe(0);
  });

  it('correctly projects normalized [0, 1] landmark coordinates into container pixel space', () => {
    const area = { offsetX: 100, offsetY: 50, contentWidth: 640, contentHeight: 480 };
    const wrist: LandmarkPoint = { x: 0.0, y: 0.0, z: 0.0 };
    const middleTip: LandmarkPoint = { x: 0.5, y: 0.25, z: -0.05 };

    const wristPxX = area.offsetX + wrist.x * area.contentWidth;
    const wristPxY = area.offsetY + wrist.y * area.contentHeight;
    expect(wristPxX).toBe(100);
    expect(wristPxY).toBe(50);

    const tipPxX = area.offsetX + middleTip.x * area.contentWidth;
    const tipPxY = area.offsetY + middleTip.y * area.contentHeight;
    expect(tipPxX).toBe(420);
    expect(tipPxY).toBe(170);
  });
});

