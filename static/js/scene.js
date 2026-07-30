(function () {
  var root = document.documentElement;
  var body = document.body;

  function setTimeOfDay(hour) {
    var h = Number(hour);
    if (Number.isNaN(h)) h = 14;
    var band = "day";
    if (h >= 6 && h <= 16) band = "day";
    else if (h >= 17 && h <= 20) band = "dusk";
    else band = "night";
    body.setAttribute("data-time", band);
    var badgeTime = document.getElementById("sceneTimeLabel");
    if (badgeTime) {
      badgeTime.textContent = String(h).padStart(2, "0") + ":00";
    }
  }

  function setWeather(weather) {
    var w = String(weather || "Clear").toLowerCase();
    var isRainy = w.indexOf("rain") !== -1;
    body.setAttribute("data-weather", isRainy ? "rainy" : "clear");
    var badgeWeather = document.getElementById("sceneWeatherLabel");
    if (badgeWeather) {
      badgeWeather.textContent = isRainy ? "Rainy" : "Sunny";
    }
    var icon = document.getElementById("sceneWeatherIcon");
    if (icon) icon.textContent = isRainy ? "\u{1F327}\uFE0F" : "\u2600\uFE0F";
  }

  window.SmartDeliveryScene = {
    setTimeOfDay: setTimeOfDay,
    setWeather: setWeather,
  };

  // Theme toggle
  var saved = localStorage.getItem("sd_theme");
  if (saved === "light" || saved === "dark") root.setAttribute("data-theme", saved);
  var themeBtn = document.getElementById("themeToggle");
  if (themeBtn) {
    themeBtn.addEventListener("click", function () {
      var current = root.getAttribute("data-theme") || "dark";
      var next = current === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      localStorage.setItem("sd_theme", next);
    });
  }

  // Mobile nav toggle
  var navToggle = document.getElementById("navToggle");
  var mainNav = document.getElementById("mainNav");
  if (navToggle && mainNav) {
    navToggle.addEventListener("click", function () {
      var isOpen = mainNav.classList.toggle("open");
      navToggle.setAttribute("aria-expanded", String(isOpen));
    });
    // Close nav on outside click (mobile only)
    document.addEventListener("click", function (e) {
      if (window.innerWidth > 640) return;
      if (!navToggle.contains(e.target) && !mainNav.contains(e.target)) {
        mainNav.classList.remove("open");
        navToggle.setAttribute("aria-expanded", "false");
      }
    });
  }

  // Active nav link highlighting
  var path = window.location.pathname;
  document.querySelectorAll(".nav-link").forEach(function (link) {
    var href = link.getAttribute("href");
    if (!href || href.indexOf("http") === 0) return;
    if (href === "/" && path === "/") link.classList.add("active");
    else if (href !== "/" && path.startsWith(href)) link.classList.add("active");
  });

  // Default scene state
  setTimeOfDay(14);
  setWeather("Clear");

  // Gentle auto weather loop on non-form pages (demo ambience)
  var hasPredictForm = document.getElementById("predictForm");
  if (!hasPredictForm) {
    var loop = ["Clear", "Rainy", "Clear"];
    var idx = 0;
    setInterval(function () {
      idx = (idx + 1) % loop.length;
      setWeather(loop[idx]);
      setTimeOfDay((new Date().getHours() + idx) % 24);
    }, 9000);
  }
})();
