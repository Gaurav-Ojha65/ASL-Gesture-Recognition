import type { HealthResponse, SystemInfoResponse } from '@/types/api';

function getApiBase(): string {
  const envBase = import.meta.env.VITE_API_BASE_URL as string | undefined;
  if (envBase && envBase.trim()) return envBase.trim().replace(/\/$/, '');
  if (typeof window !== 'undefined' && window.location?.origin) {
    return window.location.origin.replace(/\/$/, '');
  }
  return '';
}

async function fetchJson<T>(path: string): Promise<T> {
  const url = `${getApiBase()}${path}`;
  const response = await fetch(url, {
    headers: { 'Accept': 'application/json' },
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export async function fetchHealth(): Promise<HealthResponse> {
  return fetchJson<HealthResponse>('/api/health');
}

export async function fetchSystemInfo(): Promise<SystemInfoResponse> {
  return fetchJson<SystemInfoResponse>('/api/system/info');
}
