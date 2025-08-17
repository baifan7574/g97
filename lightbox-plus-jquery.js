
/*!
 * Lightbox v2.11.3
 * https://lokeshdhakar.com/projects/lightbox2/
 */
document.addEventListener("DOMContentLoaded", function () {
  const links = document.querySelectorAll('a[data-lightbox]');
  links.forEach(link => {
    link.addEventListener('click', function (e) {
      e.preventDefault();

      const overlay = document.createElement('div');
      overlay.className = 'lightboxOverlay';

      const lightbox = document.createElement('div');
      lightbox.className = 'lightbox';

      const img = document.createElement('img');
      img.className = 'lb-image';
      img.src = this.href;

      const closeBtn = document.createElement('a');
      closeBtn.className = 'lb-close';
      closeBtn.innerHTML = '&times;';
      closeBtn.href = '#';

      closeBtn.onclick = () => {
        document.body.removeChild(overlay);
        document.body.removeChild(lightbox);
        return false;
      };

      lightbox.appendChild(img);
      lightbox.appendChild(closeBtn);

      document.body.appendChild(overlay);
      document.body.appendChild(lightbox);

      overlay.style.display = 'block';
      lightbox.style.display = 'block';
    });
  });
});
