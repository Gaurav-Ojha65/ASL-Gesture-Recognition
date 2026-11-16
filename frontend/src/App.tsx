import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Sidebar } from '@/components/layout/Sidebar';
import { Dashboard } from '@/pages/Dashboard';
import { LiveTranslator } from '@/pages/LiveTranslator';
import { ModelInsights } from '@/pages/ModelInsights';
import { Diagnostics } from '@/pages/Diagnostics';
import { DevStates } from '@/pages/DevStates';
import { CameraStatusProvider, useCameraStatus } from '@/context/CameraStatusContext';

function AppShell() {
  const { cameraActive } = useCameraStatus();

  return (
    <div
      className="min-h-screen flex flex-col"
      style={{ backgroundColor: 'var(--color-bg)' }}
    >
      <Sidebar cameraActive={cameraActive} />

      <main className="flex-1">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/translator" element={<LiveTranslator />} />
          <Route path="/model" element={<ModelInsights />} />
          <Route path="/diagnostics" element={<Diagnostics />} />
          <Route path="/dev-states" element={<DevStates />} />
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
