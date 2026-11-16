import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Video,
  Brain,
  Activity,
  Hand,
  Menu,
  X,
} from 'lucide-react';
import { useState } from 'react';

interface NavItem {
  to: string;
  icon: React.ElementType;
  label: string;
}

const NAV_ITEMS: NavItem[] = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/translator', icon: Video, label: 'Live Translator' },
  { to: '/model', icon: Brain, label: 'Model Insights' },
  { to: '/diagnostics', icon: Activity, label: 'Diagnostics' },
];

export function Sidebar() {
  const [mobileOpen, setMobileOpen] = useState(false);

  const navContent = (
    <nav className="flex flex-col h-full">
      {/* Brand */}
      <div className="flex items-center gap-3 px-5 py-6 border-b border-slate-800">
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
          style={{ background: 'linear-gradient(135deg, #7c3aed, #c026d3)' }}
        >
          <Hand size={18} className="text-white" />
        </div>
        <div>
          <span className="font-semibold text-slate-100 text-sm leading-tight block">ASL Gesture</span>
          <span className="font-semibold text-slate-100 text-sm leading-tight block">Studio</span>
        </div>
      </div>

      {/* Nav links */}
      <div className="flex-1 px-3 py-4 space-y-1">
        {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all duration-150 group
              ${isActive
                ? 'bg-slate-800 text-slate-100 border-l-2 border-violet-500'
                : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/50 border-l-2 border-transparent'
              }`
            }
          >
            <Icon size={18} />
            {label}
          </NavLink>
        ))}
      </div>

      {/* Footer */}
      <div className="px-5 py-4 border-t border-slate-800">
        <p className="text-xs text-slate-600">ASL Gesture Studio v1.0</p>
      </div>
    </nav>
  );

  return (
    <>
      {/* Desktop sidebar */}
      <aside className="hidden lg:flex flex-col w-[260px] flex-shrink-0 bg-slate-900 border-r border-slate-800 h-screen sticky top-0">
        {navContent}
      </aside>

      {/* Mobile header */}
      <div className="lg:hidden fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-4 py-3 bg-slate-900 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <div
            className="w-7 h-7 rounded-md flex items-center justify-center"
            style={{ background: 'linear-gradient(135deg, #7c3aed, #c026d3)' }}
          >
            <Hand size={15} className="text-white" />
          </div>
          <span className="font-semibold text-slate-100 text-sm">ASL Gesture Studio</span>
        </div>
        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="p-2 text-slate-400 hover:text-slate-100"
          aria-label="Toggle menu"
        >
          {mobileOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
      </div>

      {/* Mobile drawer */}
      {mobileOpen && (
        <div className="lg:hidden fixed inset-0 z-40">
          <div className="absolute inset-0 bg-black/60" onClick={() => setMobileOpen(false)} />
          <aside className="relative w-[260px] h-full bg-slate-900 border-r border-slate-800">
            {navContent}
          </aside>
        </div>
      )}
    </>
  );
}
