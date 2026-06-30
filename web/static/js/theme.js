(function () {
  try {
    var stored = document.cookie.match(/(?:^|;\s*)theme=(dark|light)/);
    if (stored) {
      document.documentElement.setAttribute('data-theme', stored[1]);
    }
  } catch (e) {
   }
})();