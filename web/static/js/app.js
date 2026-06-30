(function () {
  var _sidebar = null;
  var _overlay = null;
  var _iconMenu = null;
  var _iconClose = null;
  var _isOpen = false;

  function _getEls() {
    if (!_sidebar)   _sidebar   = document.getElementById('sidebar');
    if (!_overlay)   _overlay   = document.getElementById('sidebarOverlay');
    if (!_iconMenu)  _iconMenu  = document.getElementById('iconMenu');
    if (!_iconClose) _iconClose = document.getElementById('iconClose');
  }

  function _open() {
    _getEls();
    _isOpen = true;
    if (_sidebar)   _sidebar.classList.add('open');
    if (_overlay)   _overlay.classList.add('visible');
    if (_iconMenu)  _iconMenu.style.display  = 'none';
    if (_iconClose) _iconClose.style.display = 'block';
    document.body.style.overflow = 'hidden';
  }

  function _close() {
    _getEls();
    _isOpen = false;
    if (_sidebar)   _sidebar.classList.remove('open');
    if (_overlay)   _overlay.classList.remove('visible');
    if (_iconMenu)  _iconMenu.style.display  = 'block';
    if (_iconClose) _iconClose.style.display = 'none';
    document.body.style.overflow = '';
  }

  function _toggle() {
    _getEls();
    if (_isOpen) _close(); else _open();
  }

  // Публичные функции
  window.openSidebar  = _open;
  window.closeSidebar = _close;
  window.toggleSidebar = _toggle;

  document.addEventListener('DOMContentLoaded', function () {
    _getEls();

    var btn = document.getElementById('hamburgerBtn');
    if (btn) btn.addEventListener('click', _toggle);

    if (_overlay) _overlay.addEventListener('click', _close);

    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && _isOpen) _close();
    });

    if (_sidebar) {
      _sidebar.querySelectorAll('.nav-item').forEach(function (link) {
        link.addEventListener('click', function () {
          if (window.innerWidth <= 1024) _close();
        });
      });
    }

    function _handleResize() {
      if (window.innerWidth > 1024) {
        _close();
        document.body.style.overflow = '';
      }
    }
    window.addEventListener('resize', _handleResize);
    _handleResize();
  });
})();

document.addEventListener('DOMContentLoaded', function () {

  // Flash messages auto-hide
  document.querySelectorAll('.flash').forEach(function (el) {
    setTimeout(function () { el.style.display = 'none'; }, 5000);
  });

  var dropzone      = document.getElementById('dropzone');
  var fileInput     = document.getElementById('videoInput');
  var filenameLabel = document.getElementById('selectedFilename');

  if (dropzone && fileInput) {
    dropzone.addEventListener('click', function () { fileInput.click(); });

    ['dragenter', 'dragover'].forEach(function (evt) {
      dropzone.addEventListener(evt, function (e) {
        e.preventDefault();
        dropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(function (evt) {
      dropzone.addEventListener(evt, function (e) {
        e.preventDefault();
        dropzone.classList.remove('dragover');
      });
    });

    dropzone.addEventListener('drop', function (e) {
      if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        updateFilename();
      }
    });

    fileInput.addEventListener('change', updateFilename);

    function updateFilename() {
      if (fileInput.files.length && filenameLabel) {
        filenameLabel.textContent = '✓ ' + fileInput.files[0].name;
        filenameLabel.style.display = 'block';
      }
    }
  }

  var uploadForm = document.getElementById('uploadForm');
  if (uploadForm) {
    uploadForm.addEventListener('submit', function () {
      var submitBtn = uploadForm.querySelector('button[type="submit"]');
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.textContent = submitBtn.dataset.loadingText || submitBtn.textContent;
      }
    });
  }

  // ── Processing page polling ──────────────────────────────────────────
  var statusRoot = document.getElementById('statusRoot');
  if (statusRoot) {
    var analysisId = statusRoot.dataset.analysisId;
    var totalSteps = parseInt(statusRoot.dataset.totalSteps || '6', 10);

    var poll = function () {
      fetch('/api/status/' + analysisId)
        .then(function (r) { return r.json(); })
        .then(function (data) {
          if (data.status === 'error') { showError(data.error); return; }
          if (data.status === 'cancelled') { window.location.href = '/cases'; return; }
          updateSteps(data.current_step, totalSteps);
          if (data.done && data.redirect) { window.location.href = data.redirect; return; }
          setTimeout(poll, 3000);
        })
        .catch(function () { setTimeout(poll, 5000); });
    };

    function updateSteps(currentStep, total) {
      for (var i = 1; i <= total; i++) {
        var el = document.querySelector('.processing-step[data-step="' + i + '"]');
        if (!el) continue;
        el.classList.remove('is-done', 'is-active');
        if (i < currentStep) el.classList.add('is-done');
        else if (i === currentStep) el.classList.add('is-active');
      }
    }

    function showError(message) {
      var errorBox = document.getElementById('processingError');
      var stepsBox = document.getElementById('processingSteps');
      if (stepsBox) stepsBox.style.display = 'none';
      if (errorBox) {
        errorBox.style.display = 'block';
        var msgEl = errorBox.querySelector('.processing-error-msg');
        if (msgEl && message) msgEl.textContent = message;
      }
    }

    setTimeout(poll, 1500);
  }

});