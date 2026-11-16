import { Link } from 'react-router-dom';

export function NotFound() {
  return (
    <div className="min-h-[70vh] flex flex-col items-center justify-center text-center px-5 py-16">
      <div className="bg-coral-soft text-coral-text text-xs font-semibold px-3 py-1 rounded-pill uppercase tracking-wider mb-4">
        404
      </div>
      <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-ink mb-3">
        Page not found
      </h1>
      <p className="text-muted text-base max-w-md mb-8 leading-relaxed">
        The page you are looking for doesn't exist or has been moved. Check the URL or return home.
      </p>
      <Link
        to="/"
        className="inline-flex items-center justify-center px-6 py-3 bg-brand text-on-brand font-semibold rounded-button hover:opacity-90 transition-opacity no-underline"
      >
        Return to Home
      </Link>
    </div>
  );
}
