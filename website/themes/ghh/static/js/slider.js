/* Before/after comparison slider */
document.querySelectorAll('.comparison-slider').forEach(slider => {
  const afterWrap = slider.querySelector('.after-wrap');
  const handle = slider.querySelector('.slider-handle');
  if (!afterWrap || !handle) return;

  const afterImg = afterWrap.querySelector('img');
  let dragging = false;

  function setPosition(x) {
    const rect = slider.getBoundingClientRect();
    let pct = ((x - rect.left) / rect.width) * 100;
    pct = Math.max(2, Math.min(98, pct));
    afterWrap.style.width = pct + '%';
    handle.style.left = pct + '%';
    if (afterImg) {
      afterImg.style.width = (100 / pct * 100) + '%';
    }
  }

  function onStart(e) {
    e.preventDefault();
    dragging = true;
    const x = e.touches ? e.touches[0].clientX : e.clientX;
    setPosition(x);
  }

  function onMove(e) {
    if (!dragging) return;
    const x = e.touches ? e.touches[0].clientX : e.clientX;
    setPosition(x);
  }

  function onEnd() { dragging = false; }

  slider.addEventListener('mousedown', onStart);
  slider.addEventListener('touchstart', onStart, { passive: false });
  window.addEventListener('mousemove', onMove);
  window.addEventListener('touchmove', onMove, { passive: false });
  window.addEventListener('mouseup', onEnd);
  window.addEventListener('touchend', onEnd);

  /* Initialize after image width */
  setPosition(slider.getBoundingClientRect().left + slider.getBoundingClientRect().width / 2);
});

/* Smooth scroll for nav links */
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    const target = document.querySelector(a.getAttribute('href'));
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});
