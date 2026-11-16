import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Sidebar } from '@/components/layout/Sidebar';
import { Dashboard } from '@/pages/Dashboard';
import { LiveTranslator } from '@/pages/LiveTranslator';
import { ModelInsights } from '@/pages/ModelInsights';
import { Diagnostics } from '@/pages/Diagnostics';

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex min-h-screen bg-slate-950">
        <Sidebar />
        <main className="flex-1 min-w-0 lg:overflow-y-auto">
          {/* Mobile top-padding for fixed header */}
          <div className="lg:hidden h-14" />
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/translator" element={<LiveTranslator />} />
            <Route path="/model" element={<ModelInsights />} />
            <Route path="/diagnostics" element={<Diagnostics />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
