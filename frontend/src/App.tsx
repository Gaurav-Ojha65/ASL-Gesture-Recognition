import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Navbar } from '@/components/layout/Navbar';
import { Dashboard } from '@/pages/Dashboard';
import { LiveTranslator } from '@/pages/LiveTranslator';
import { ModelInsights } from '@/pages/ModelInsights';
import { Diagnostics } from '@/pages/Diagnostics';
import { NotFound } from '@/pages/NotFound';
import { CameraStatusProvider, useCameraStatus } from '@/context/CameraStatusContext';

const DevStates = import.meta.env.DEV
  ? lazy(() => import('@/pages/DevStates').then((m) => ({ default: m.DevStates })))
  : null;

function AppShell() {
  const { cameraActive } = useCameraStatus();

  return (
    <div
      className="min-h-screen flex flex-col"
      style={{ backgroundColor: 'var(--color-bg)' }}
    >
      <Navbar cameraActive={cameraActive} />

      <main className="flex-1">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/translator" element={<LiveTranslator />} />
          <Route path="/model" element={<ModelInsights />} />
          <Route path="/diagnostics" element={<Diagnostics />} />
          {import.meta.env.DEV && DevStates && (
            <Route
              path="/dev-states"
              element={
                <Suspense fallback={null}>
                  <DevStates />
                </Suspense>
              }
            />
          )}
          {/* Catch-all route for unmatched paths */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>

      <footer
        className="text-center text-sm py-4"
        style={{ color: 'var(--color-muted)' }}
      >
        ASL Gesture Studio · v1.0
      </footer>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <CameraStatusProvider>
        <AppShell />
      </CameraStatusProvider>
    </BrowserRouter>
  );
}
