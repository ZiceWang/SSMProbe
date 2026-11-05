import katex from 'katex';

export function MathFormula({ children, display = true }: { children: string; display?: boolean }) {
  const html = katex.renderToString(children, { displayMode: display, throwOnError: true, strict: 'error' });
  return <span className={display ? 'math-display' : undefined} dangerouslySetInnerHTML={{ __html: html }} />;
}
