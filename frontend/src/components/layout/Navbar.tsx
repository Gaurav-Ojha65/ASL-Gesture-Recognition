import { NavLink } from 'react-router-dom';
import { Logo } from '@/components/brand/Logo';

interface NavBarProps {
  cameraActive?: boolean;
}

const NAV_ITEMS = [
  { to: '/', label: 'Home', end: true },
  { to: '/translator', label: 'Translate', end: false },
  { to: '/model', label: 'How it works', end: false },
  { to: '/diagnostics', label: 'Developer tools', end: false },
];

export function Navbar({ cameraActive = false }: NavBarProps) {
  return (
    <header
      className="sticky top-0 z-40 w-full bg-surface border-b"
      style={{ borderColor: 'var(--color-line)' }}
    >
      <div
        className="mx-auto flex items-center justify-between gap-3 md:gap-6 px-4 md:px-10 h-14"
        style={{ maxWidth: '1240px' }}
      >
        {/* Brand */}
        <NavLink
          to="/"
          className="flex items-center gap-2.5 flex-shrink-0 no-underline btn-sm"
          aria-label="ASL Gesture Studio home"
        >
          <Logo size={36} />
          <span
            className="font-bold text-sm leading-tight hidden xs:inline-block sm:inline-block"
            style={{ color: 'var(--color-ink)' }}
          >
            ASL Gesture<br />Studio
          </span>
        </NavLink>

        {/* Nav links */}
        <nav
          className="flex-1 flex items-center justify-start md:justify-center gap-1 overflow-x-auto py-1 no-scrollbar"
          aria-label="Main navigation"
        >
          {NAV_ITEMS.map(({ to, label, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                `btn-sm flex-shrink-0 flex items-center px-3.5 py-1.5 rounded-pill text-sm font-medium transition-colors no-underline
                ${isActive
                  ? 'bg-brand-soft text-brand'
                  : 'text-muted hover:text-ink hover:bg-track'
                }`
              }
            >
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Camera status pill */}
        <div
          className="flex-shrink-0 hidden sm:flex items-center gap-1.5 text-sm font-medium btn-sm px-2 md:px-3"
          style={{ color: cameraActive ? 'var(--color-success)' : 'var(--color-muted)' }}
          aria-live="polite"
          aria-label={cameraActive ? 'Camera on' : 'Camera off'}
        >
          <span
            className="w-2 h-2 rounded-full flex-shrink-0"
            style={{
              backgroundColor: cameraActive ? 'var(--color-success)' : 'var(--color-sand)',
            }}
            aria-hidden="true"
          />
          {cameraActive ? 'Camera on' : 'Camera off'}
        </div>
      </div>
    </header>
  );
}

export const Sidebar = Navbar;
