import { useState } from 'react';
import { project, imageNetResults, routeGeometry } from './content/project';
import { FigureGallery } from './components/FigureGallery';
import { Icon } from './components/Icon';
import { MathFormula } from './components/Math';

const asset = (path: string) => `${import.meta.env.BASE_URL}${path}`;

export default function App() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState(false);

  async function copyCitation() {
    try {
      await navigator.clipboard.writeText(project.bibtex);
      setCopied(true);
      setCopyError(false);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopyError(true);
    }
  }

  return <>
    <a className="skip-link" href="#main">Skip to content</a>
    <nav className="navbar" role="navigation" aria-label="Main navigation">
      <div className="navbar-brand"><button className={`navbar-burger ${menuOpen ? 'is-active' : ''}`} aria-label="Toggle navigation" aria-expanded={menuOpen} aria-controls="project-menu" onClick={() => setMenuOpen(!menuOpen)}><span aria-hidden="true" /><span aria-hidden="true" /><span aria-hidden="true" /></button></div>
      <div id="project-menu" className={`navbar-menu ${menuOpen ? 'is-active' : ''}`}>
        <div className="navbar-start nav-center">
          <a className="navbar-item" href="#" aria-label="Back to top" onClick={() => setMenuOpen(false)}><span className="icon"><Icon name="home" /></span></a>
          <div className="navbar-item has-dropdown is-hoverable">
            <button className="navbar-link project-nav-label" onClick={() => setDropdownOpen(!dropdownOpen)} aria-expanded={dropdownOpen}>SSMProbe</button>
            <div className={`navbar-dropdown ${dropdownOpen ? 'is-open' : ''}`}>
              {[['Abstract', 'abstract'], ['Method', 'method'], ['Results', 'results'], ['BibTeX', 'bibtex']].map(([label, id]) => <a key={id} className="navbar-item" href={`#${id}`} onClick={() => { setMenuOpen(false); setDropdownOpen(false); }}>{label}</a>)}
            </div>
          </div>
        </div>
      </div>
    </nav>

    <main id="main">
      <section className="hero"><div className="hero-body"><div className="container is-max-desktop"><div className="columns is-centered"><div className="column has-text-centered">
        <h1 className="title is-1 publication-title">{project.titleLines[0]}<br className="title-break" /> {project.titleLines[1]}</h1>
        <div className="is-size-5 publication-authors">{project.authors.map((author, index) => <span className="author-block" key={author.name}>{index > 0 && ', '}{author.href ? <a href={author.href} target="_blank" rel="noreferrer">{author.name}</a> : <span className="author-name">{author.name}</span>}<sup>{author.affiliation}</sup></span>)}</div>
        <div className="is-size-5 publication-authors affiliations">{project.affiliations.map((name, index) => <span className="author-block" key={name}><sup>{index + 1}</sup>{name}</span>)}</div>
        <div className="is-size-5 publication-venue"><a href="https://neurips.cc/virtual/2026/poster/156006" target="_blank" rel="noreferrer">NeurIPS 2026</a></div>
        <div className="column has-text-centered"><div className="publication-links">{project.links.map((link) => <span className="link-block" key={link.label}><a className="external-link button is-normal is-rounded is-dark" href={link.href} target="_blank" rel="noreferrer"><span className="icon"><Icon name={link.icon} /></span><span>{link.label}</span></a></span>)}</div></div>
      </div></div></div></div></section>

      <section className="hero teaser"><div className="container is-max-desktop"><div className="hero-body">
        <img className="teaser-image" src={asset('figures/evidence-routes.png')} alt="The same CUB bird image and the learned evidence-route overlays for MAE, BEiT, supervised ViT, and DINOv2. Color encodes normalized position from early to late in the sequence." width="1920" height="490" fetchPriority="high" />
        <h2 className="subtitle has-text-centered"><span className="project-name">SSMProbe</span> learns where patch evidence goes,<br className="desktop-break" /> and exposes how a sequential readout uses it.</h2>
      </div></div></section>

      <FigureGallery />

      <section className="section" id="abstract"><div className="container is-max-desktop"><div className="columns is-centered has-text-centered"><div className="column is-four-fifths">
        <h2 className="title is-3">Abstract</h2>
        <div className="content has-text-justified">
          <p>Probing frozen vision transformers typically uses permutation-invariant aggregation (GAP or <code>[CLS]</code>), treating patch tokens as an unstructured set. Content-dependent probes such as self-attention are useful accuracy controls, but they do not expose a fixed token schedule or fixed position weights for auditing.</p>
          <p>We introduce <strong>SSMProbe</strong>, a probe of frozen vision transformers that learns a Sinkhorn patch route and reads it with a diagonal S4 decoder. The decoder’s causal linear time-invariant (LTI) recurrence yields fixed position coefficients, exposing how patch arrangement interacts with sequential weighting. Held-out spatial predictability measures route organization beyond a location-bias null. On CUB and Cars, MAE, BEiT, DINOv2, and supervised ViT yield spatially organized routes with distinct readout profiles: DINOv2 is <code>[CLS]</code>-dominant, while MAE benefits from learned routing over fixed scans. On MAE/ImageNet, Sinkhorn reaches 70.33% accuracy, 5.55 percentage points above the strongest fixed scan.</p>
        </div>
      </div></div></div></section>

      <section className="section" id="method"><div className="container is-max-desktop">
        <h2 className="title is-3">Learning an evidence route</h2>
        <div className="content has-text-justified"><p>A frozen vision transformer supplies patch tokens. A lightweight scorer assigns a scalar score to each token, and a differentiable Sinkhorn assignment routes the patches into a sequence. A diagonal S4 decoder reads the sequence and predicts the class from its final output.</p></div>
        <div className="math-pipeline"><MathFormula display={false}>{String.raw`\widetilde{T}=P^{\top}T,`}</MathFormula><MathFormula display={false}>{String.raw`Z=\mathrm{S4}(\widetilde{T}),`}</MathFormula><MathFormula display={false}>{String.raw`\widehat{y}=WZ_N+b`}</MathFormula></div>
        <div className="columns method-columns">
          <div className="column"><div className="content"><h3 className="title is-4">Where is the evidence?</h3><p>The learned route assigns each source patch an expected position. We measure its spatial organization using <strong>ERSI</strong>: held-out code length saved by a spatial route model over a location-bias null, in bits per routed patch.</p><p>Positive ERSI indicates spatial organization beyond marginal location preferences.</p></div></div>
          <div className="column"><div className="content"><h3 className="title is-4">How is it read?</h3><p>After training, S4 supplies fixed coefficients for the routed positions. This exposes a decomposition of the logits and allows alternative patch schedules to be evaluated analytically.</p><MathFormula>{String.raw`f(T,\pi)=\sum_{k=1}^{N}\alpha_k\,t_{\pi(k)}+b`}</MathFormula></div></div>
        </div>
      </div></section>

      <section className="section" id="results"><div className="container is-max-desktop">
        <h2 className="title is-3">Learned routes improve MAE readout</h2>
        <div className="content"><p>On frozen MAE / ImageNet-1K, Sinkhorn + S4 reaches <strong>70.33%</strong> top-1 accuracy, a gain of <strong>5.55 percentage points</strong> over the strongest fixed scan. The comparison below follows the paper’s five-epoch protocol.</p></div>
        <div className="table-container"><table className="booktabs results-table"><caption>Selected readout comparisons on ImageNet-1K. Mean ± sample standard deviation over five seeds; top-1 accuracy (%).</caption><thead><tr><th scope="col">Readout</th><th scope="col">Top-1 accuracy (%)</th></tr></thead><tbody>{imageNetResults.map(([method, accuracy], index) => <tr key={method} className={index === imageNetResults.length - 1 ? 'our-method' : undefined}><th scope="row">{method}</th><td>{accuracy}</td></tr>)}</tbody></table></div>
        <p className="figure-caption">Selected methods from v3, Table 1. The complete table, including decoder controls and the wider Transformer reference, is in the <a href={project.links[0].href} target="_blank" rel="noreferrer">paper</a>.</p>
      </div></section>

      <section className="section"><div className="container is-max-desktop">
        <h2 className="title is-3">Spatially organized routes across backbones</h2>
        <div className="content"><p>All four frozen backbones yield positive ERSI on CUB and Cars. Their readout profiles differ: MAE is routing-sensitive; BEiT and supervised ViT support compact patch evidence; DINOv2 combines organized patch routes with a <code>[CLS]</code>-dominant readout.</p></div>
        <figure className="paper-figure"><img src={asset('figures/route-attention-2x4.png')} alt="Updated two-row, four-column comparison: route position above attention rollout, with columns for MAE, BEiT, ViT, and DINOv2." loading="lazy" width="1950" height="1020" /><figcaption>Updated route-position and attention-rollout comparison on a shared CUB image. Columns show MAE, BEiT, ViT, and DINOv2.</figcaption></figure>
        <div className="table-container"><table className="booktabs geometry-table"><caption>Held-out spatial information, ERSI₂₀ (bits per routed patch).</caption><thead><tr><th scope="col">Backbone</th><th scope="col">CUB</th><th scope="col">Cars</th></tr></thead><tbody>{routeGeometry.map(([backbone, cub, cars]) => <tr key={backbone}><th scope="row">{backbone}</th><td>{cub}</td><td>{cars}</td></tr>)}</tbody></table></div>
      </div></section>

      <section className="section" id="scope"><div className="container is-max-desktop"><div className="content"><h2 className="title is-3">Scope and interpretation</h2><p>The learned routes reflect the frozen features, downstream supervision, and decoder coefficients jointly. Evaluation covers classification on ImageNet-1K, CUB, and Cars, with object-mask alignment on CUB. Route geometry depends on the trained readout, and extension to dense prediction requires task-specific evaluation. The paper reports decoder controls, scorer transfer, and input-sensitivity analyses.</p></div></div></section>

      <section className="section" id="bibtex"><div className="container is-max-desktop content"><div className="citation-header"><h2 className="title is-3">BibTeX</h2><button className="button is-small" onClick={copyCitation}>{copied ? 'Copied!' : 'Copy'}</button></div><pre><code>{project.bibtex}</code></pre><p className="copy-status" role="status">{copyError ? 'Copy is unavailable in this browser. Select the citation above to copy it.' : copied ? 'Citation copied to clipboard.' : ''}</p></div></section>
    </main>

    <footer className="footer"><div className="container"><div className="content has-text-centered footer-icons"><a href={project.links[0].href} target="_blank" rel="noreferrer" aria-label="Read the paper"><Icon name="paper" /></a><a href={project.links[2].href} target="_blank" rel="noreferrer" aria-label="View source code"><Icon name="github" /></a></div><div className="columns is-centered"><div className="column is-8"><div className="content"><p>Website adapted from <a href="https://nerfies.github.io/" target="_blank" rel="noreferrer">Nerfies</a>, rebuilt with React and Vite. Website template licensed under <a href="https://creativecommons.org/licenses/by-sa/4.0/" target="_blank" rel="noreferrer">CC BY-SA 4.0</a>.</p></div></div></div></div></footer>
  </>;
}
