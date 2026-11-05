export const project = {
  name: 'SSMProbe',
  title: 'Learning Dynamic Evidence Routes for Vision Transformer Probing',
  titleLines: ['Learning Dynamic Evidence Routes', 'for Vision Transformer Probing'],
  authors: [
    { name: 'Zice Wang', affiliation: '1', href: 'https://github.com/ZiceWang' },
    { name: 'Zhenyu Zhang', affiliation: '2', href: undefined },
  ],
  affiliations: [
    'School of Future Technology, Northeastern University',
    'School of Computer Science and Technology, Beijing Institute of Technology',
  ],
  links: [
    { label: 'Paper', icon: 'paper', href: 'https://arxiv.org/pdf/2605.00915v3' },
    { label: 'arXiv', icon: 'arxiv', href: 'https://arxiv.org/abs/2605.00915v3' },
    { label: 'Code', icon: 'github', href: 'https://github.com/ZiceWang/SSMProbe' },
  ],
  tagline: 'SSMProbe learns where patch evidence goes, and exposes how a sequential readout uses it.',
  bibtex: `@article{wang2026learning,
  title   = {Learning Dynamic Evidence Routes for Vision Transformer Probing},
  author  = {Wang, Zice and Zhang, Zhenyu},
  journal = {arXiv preprint arXiv:2605.00915},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.00915v3}
}`,
};

export const figures = [
  {
    src: 'figures/route-attention-2x4.png',
    title: 'Routes and attention',
    alt: 'CUB image, learned evidence routes, and attention rollouts for MAE, BEiT, ViT, and DINOv2.',
    caption: 'Updated route-position and attention-rollout comparison on a shared CUB image. Columns show MAE, BEiT, ViT, and DINOv2.',
  },
  {
    src: 'figures/token-sensitivity.png',
    title: 'Token sensitivity',
    alt: 'Token-drop sensitivity curves from the v3 paper comparing patch removal interventions.',
    caption: 'Source-aligned token-drop interventions measure how classification changes when evidence patches are removed.',
  },
  {
    src: 'figures/permutation-matrices.png',
    title: 'Learned assignments',
    alt: 'Sinkhorn patch-to-position assignment matrices across frozen vision transformer backbones.',
    caption: 'Sinkhorn assignment matrices map source patch tokens to positions in the learned sequence.',
  },
];

export const imageNetResults = [
  ['GAP', '58.10 ± 0.03'],
  ['CLS', '56.65 ± 0.03'],
  ['Raster scan + S4', '63.64 ± 0.05'],
  ['4-dir VMamba-style + S4', '64.68 ± 0.07'],
  ['4-dir Snake + S4', '64.76 ± 0.06'],
  ['4-dir Diagonal + S4', '64.78 ± 0.07'],
  ['Random-dynamic permutation + S4', '63.95 ± 0.06'],
  ['Attention Pool', '67.70 ± 0.04'],
  ['Content-Weighted Pool', '67.68 ± 0.06'],
  ['Sinkhorn + S4 (SSMProbe)', '70.33 ± 0.05'],
];

export const routeGeometry = [
  ['MAE', '0.318', '0.336'],
  ['BEiT', '0.175', '0.276'],
  ['ViT', '0.113', '0.296'],
  ['DINOv2', '0.484', '0.328'],
];
