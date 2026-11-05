export function Icon({ name }: { name: string }) {
  if (name === 'arxiv') return <span className="arxiv-icon" aria-hidden="true">α</span>;
  const paths: Record<string, React.ReactNode> = {
    home: <path d="m12 3 10 8h-3v10h-5v-6h-4v6H5V11H2L12 3Z" />,
    paper: <><path d="M5 2h9l5 5v15H5V2Zm9 1v5h5M8 17l2-6 3 7m-5-3h7" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" /></>,
    github: <path d="M12 .9a11.1 11.1 0 0 0-3.51 21.63c.55.1.76-.24.76-.54v-2.07c-3.09.67-3.74-1.31-3.74-1.31-.5-1.28-1.23-1.62-1.23-1.62-1.01-.7.08-.68.08-.68 1.12.08 1.71 1.15 1.71 1.15.99 1.7 2.59 1.21 3.22.92.1-.72.39-1.21.7-1.49-2.47-.28-5.06-1.24-5.06-5.49 0-1.21.43-2.2 1.14-2.97-.11-.28-.5-1.4.11-2.91 0 0 .93-.3 3.05 1.13A10.6 10.6 0 0 1 12 6.37c.95 0 1.9.13 2.79.38 2.12-1.44 3.04-1.13 3.04-1.13.61 1.51.23 2.63.11 2.91.71.77 1.14 1.76 1.14 2.97 0 4.26-2.59 5.2-5.07 5.48.4.35.75 1.02.75 2.06V22c0 .3.2.65.77.54A11.1 11.1 0 0 0 12 .9Z" />,
    left: <path d="m15 5-7 7 7 7" fill="none" stroke="currentColor" strokeWidth="2" />,
    right: <path d="m9 5 7 7-7 7" fill="none" stroke="currentColor" strokeWidth="2" />,
    close: <path d="m6 6 12 12M18 6 6 18" fill="none" stroke="currentColor" strokeWidth="2" />,
  };
  return <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" focusable="false">{paths[name]}</svg>;
}
