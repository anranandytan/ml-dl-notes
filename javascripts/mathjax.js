/* mkdocs-material requires a MathJax config script loaded before mathjax itself */
window.MathJax = {
  tex: {
    inlineMath:  [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    packages: { "[+]": ["boldsymbol"] },
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

/* pymdownx.arithmatex emits $...$ and $$...$$ as-is;
   tell MathJax to also handle those delimiters */
document$.subscribe(() => {
  MathJax.typesetPromise?.();
});
