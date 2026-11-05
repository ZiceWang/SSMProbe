# SSMProbe project page

A clean React + TypeScript + Vite implementation of the original [Nerfies](https://nerfies.github.io/) academic project-page design. The page uses the original Bulma stylesheet, the same font families and typography, centered authors, dark rounded resource buttons, and the original section proportions.

Project homepage: **https://zicewang.github.io/SSMProbe/**

## Develop

Requires Node.js 22.12+ or 24+.

```sh
npm ci
npm run dev
```

Open the Local URL printed by Vite. To select a port directly on PowerShell:

```powershell
node node_modules/vite/bin/vite.js --host 127.0.0.1 --port 5186 --strictPort
```

## Build

```sh
npm run build
npm run preview
```

`dist/` is the deployable static website. The Vite base is `/SSMProbe/`, matching the GitHub Pages project URL.

## Edit content

- `src/content/project.ts`: title, authors, institutions, resource links, citation, and numerical tables.
- `src/App.tsx`: abstract, method, results, and section arrangement.
- `src/styles/index.css`: Nerfies layout and responsive adjustments.
- `public/figures/`: rendered figures from arXiv 2605.00915v3 and the author-provided updated two-row comparison; provenance is recorded in `public/figures/provenance.json`.
- `src/assets/fonts/`: locally hosted original font families; licenses are in `licenses/`.

The figure viewer supports keyboard navigation and Escape to close. The citation has a Copy button. The page contains no analytics, original Nerfies demonstration media, jQuery, or interpolation preloading.

## Publishing

The `feat/project-page` branch contains the React source. GitHub Pages serves the compiled files from the root of the separate `docs/pages` branch. Build with `npm run build`, then replace that branch with the contents of `dist/` to publish an update. The `.nojekyll` file bypasses Jekyll processing. Build artifacts are not committed to the source branch.

## Attribution

Layout and original Bulma styling adapted from [nerfies/nerfies.github.io](https://github.com/nerfies/nerfies.github.io), under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Preserve the visible footer attribution. This license applies to the adapted website template; the paper and research assets retain their respective rights.

Bulma is MIT licensed. Font licenses are included in `licenses/`. KaTeX is MIT licensed.
