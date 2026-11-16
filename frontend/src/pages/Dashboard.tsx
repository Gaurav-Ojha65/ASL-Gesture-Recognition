import { Link } from 'react-router-dom';

const SAMPLE_TILES = [
  'A', 'B', 'C', 'D',
  'E', 'F', 'G', 'H',
  'I', 'K', 'L', 'M',
];

export function Dashboard() {
  return (
    <div className="max-w-[1240px] mx-auto px-5 md:px-10 py-10 md:py-16 flex flex-col gap-12">
      {/* ── Hero Section ──────────────────────────────────────────────── */}
      <section className="grid grid-cols-1 lg:grid-cols-[1.1fr_0.9fr] gap-10 items-center">
        {/* Left: Text & CTA */}
        <div className="flex flex-col items-start gap-6">
          <span
            className="inline-block text-xs font-semibold px-3 py-1 rounded-pill"
            style={{
              backgroundColor: 'var(--color-coral-soft)',
              color: 'var(--color-coral-text)',
            }}
          >
            American Sign Language, A to Z
          </span>

          <h1
            className="text-4xl sm:text-5xl lg:text-[60px] font-bold tracking-tight"
            style={{
              color: 'var(--color-ink)',
              lineHeight: 1.05,
              letterSpacing: '-0.03em',
            }}
          >
            Show a sign.<br />See it as text.
          </h1>

          <p
            className="text-base sm:text-lg max-w-[480px] leading-relaxed"
            style={{ color: 'var(--color-muted)' }}
          >
            Hold a letter of the ASL alphabet up to your camera and watch it appear
            as words, with the option to hear them read aloud.
          </p>

          <div className="flex flex-wrap items-center gap-4 pt-2">
            <Link
              to="/translator"
              className="btn-sm flex items-center justify-center font-semibold px-6 py-3 transition-colors no-underline"
              style={{
                backgroundColor: 'var(--color-brand)',
                color: 'var(--color-on-brand)',
                borderRadius: 'var(--radius-button)',
                minHeight: '44px',
              }}
            >
              Start translating
            </Link>

            <Link
              to="/model"
              className="btn-sm flex items-center justify-center font-medium px-6 py-3 border transition-colors no-underline"
              style={{
                backgroundColor: 'var(--color-surface)',
                borderColor: 'var(--color-line)',
                color: 'var(--color-ink)',
                borderRadius: 'var(--radius-button)',
                minHeight: '44px',
              }}
            >
              How it works
            </Link>
          </div>
        </div>

        {/* Right: Signs It Understands Card */}
        <div
          className="bg-surface border p-6 sm:p-8 flex flex-col gap-5 w-full max-w-[480px] mx-auto lg:ml-auto"
          style={{
            borderColor: 'var(--color-line)',
            borderRadius: 'var(--radius-card)',
          }}
        >
          <div className="flex items-center justify-between">
            <span
              className="text-xs font-semibold uppercase tracking-widest"
              style={{ color: 'var(--color-muted)', letterSpacing: '0.06em' }}
            >
              Signs it understands
            </span>
            <span
              className="text-xs font-medium"
              style={{ color: 'var(--color-muted)' }}
            >
              28 in total
            </span>
          </div>

          {/* 4x3 Grid */}
          <div className="grid grid-cols-4 gap-3">
            {SAMPLE_TILES.map((letter) => {
              const isHighlight = letter === 'F';
              return (
                <div
                  key={letter}
                  className="aspect-square flex items-center justify-center text-xl font-bold transition-transform"
                  style={{
                    backgroundColor: isHighlight
                      ? 'var(--color-brand-soft)'
                      : 'var(--color-track)',
                    border: isHighlight
                      ? '2px solid var(--color-brand)'
                      : '1px solid transparent',
                    color: isHighlight
                      ? 'var(--color-brand)'
                      : 'var(--color-ink)',
                    borderRadius: '16px',
                  }}
                >
                  {letter}
                </div>
              );
            })}
          </div>

          <p
            className="text-xs font-medium"
            style={{ color: 'var(--color-muted)' }}
          >
            A to Z, plus space and delete.
          </p>
        </div>
      </section>

      {/* ── Feature Cards ─────────────────────────────────────────────── */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-4">
        {/* Feature 1 */}
        <div
          className="bg-surface border p-6 sm:p-7 flex flex-col gap-2.5"
          style={{
            borderColor: 'var(--color-line)',
            borderRadius: 'var(--radius-card)',
          }}
        >
          <h3
            className="text-lg font-bold"
            style={{ color: 'var(--color-ink)' }}
          >
            Works with your camera
          </h3>
          <p
            className="text-sm leading-relaxed"
            style={{ color: 'var(--color-muted)' }}
          >
            Your hand is tracked through your webcam. Nothing to install and no gloves or sensors.
          </p>
        </div>

        {/* Feature 2 */}
        <div
          className="bg-surface border p-6 sm:p-7 flex flex-col gap-2.5"
          style={{
            borderColor: 'var(--color-line)',
            borderRadius: 'var(--radius-card)',
          }}
        >
          <h3
            className="text-lg font-bold"
            style={{ color: 'var(--color-ink)' }}
          >
            Confirms before it types
          </h3>
          <p
            className="text-sm leading-relaxed"
            style={{ color: 'var(--color-muted)' }}
          >
            A sign only counts after you hold it steady for a moment, so passing gestures don't create stray letters.
          </p>
        </div>

        {/* Feature 3 */}
        <div
          className="bg-surface border p-6 sm:p-7 flex flex-col gap-2.5"
          style={{
            borderColor: 'var(--color-line)',
            borderRadius: 'var(--radius-card)',
          }}
        >
          <h3
            className="text-lg font-bold"
            style={{ color: 'var(--color-ink)' }}
          >
            Honest about limits
          </h3>
          <p
            className="text-sm leading-relaxed"
            style={{ color: 'var(--color-muted)' }}
          >
            J and Z need motion, so they can be misread. Light, distance and hand angle affect results.
          </p>
        </div>
      </section>
    </div>
  );
}
