import { useEffect, useRef, useState } from 'react';
import { figures } from '../content/project';
import { Icon } from './Icon';

const asset = (path: string) => `${import.meta.env.BASE_URL}${path}`;

export function FigureGallery() {
  const [selected, setSelected] = useState<number | null>(null);
  const dialog = useRef<HTMLDialogElement>(null);
  const closeButton = useRef<HTMLButtonElement>(null);
  const lastTrigger = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (selected !== null) {
      dialog.current?.showModal();
      closeButton.current?.focus();
    } else {
      dialog.current?.close();
      lastTrigger.current?.focus();
    }
  }, [selected]);

  const close = () => setSelected(null);
  const step = (delta: number) => setSelected((current) => current === null ? null : (current + delta + figures.length) % figures.length);

  return <>
    <section className="hero is-light is-small gallery-section" aria-label="Research figures">
      <div className="hero-body"><div className="container gallery-container"><div className="results-carousel">
        {figures.map((figure, index) => <button key={figure.src} className="figure-preview" onClick={(event) => { lastTrigger.current = event.currentTarget; setSelected(index); }} aria-label={`Enlarge ${figure.title.toLowerCase()}`}>
          <img src={asset(figure.src)} alt={figure.alt} loading="lazy" width="720" height="240" />
          <span>{figure.title}</span>
        </button>)}
      </div></div></div>
    </section>
    <dialog ref={dialog} className="figure-dialog" aria-labelledby="figure-title" onCancel={close} onClose={() => setSelected(null)} onClick={(event) => { if (event.target === event.currentTarget) close(); }} onKeyDown={(event) => { if (event.key === 'ArrowLeft') step(-1); if (event.key === 'ArrowRight') step(1); }}>
      {selected !== null && <div className="figure-dialog-content">
        <div className="figure-dialog-header"><h2 id="figure-title">{figures[selected].title}</h2><button ref={closeButton} className="icon-button" onClick={close} aria-label="Close figure"><Icon name="close" /></button></div>
        <img src={asset(figures[selected].src)} alt={figures[selected].alt} />
        <p>{figures[selected].caption}</p>
        <div className="figure-dialog-controls"><button className="button is-small" onClick={() => step(-1)}><Icon name="left" /> Previous</button><span>{selected + 1} / {figures.length}</span><button className="button is-small" onClick={() => step(1)}>Next <Icon name="right" /></button></div>
      </div>}
    </dialog>
  </>;
}
