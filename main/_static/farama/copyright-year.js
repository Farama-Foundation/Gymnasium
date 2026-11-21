(() => {
  const year = new Date().getFullYear();
  const noticeYear = /©(?:\s*\d{4}(?:\s*[-–—]\s*\d{4})?\b)?/u;

  for (const notice of document.querySelectorAll(".bottom-of-page .copyright")) {
    // Update the text without replacing the optional copyright link.
    const text = Array.from(notice.childNodes).find(
      (node) => node.nodeType === Node.TEXT_NODE && noticeYear.test(node.textContent),
    );
    if (text) {
      text.textContent = text.textContent.replace(noticeYear, `© ${year}`);
    }
  }
})();
