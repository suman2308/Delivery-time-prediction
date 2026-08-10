/* ═══════════════════════════════════════════════════════════════════════════
   Smart Courier Prediction — Application Script
   Theme, navigation, dropdowns, scroll reveal, counters, tabs, modals,
   toasts, and page-specific interactions.
   ═══════════════════════════════════════════════════════════════════════════ */
(function () {
  "use strict";

  var root = document.documentElement;

  /* ── Theme ─────────────────────────────────────────────────────────────── */
  (function initTheme() {
    var saved = null;
    try { saved = localStorage.getItem("scp_theme"); } catch (e) { /* ignore */ }
    if (saved === "light" || saved === "dark") root.setAttribute("data-theme", saved);

    var btn = document.getElementById("themeToggle");
    if (btn) {
      btn.addEventListener("click", function () {
        var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
        root.setAttribute("data-theme", next);
        try { localStorage.setItem("scp_theme", next); } catch (e) { /* ignore */ }
      });
    }
  })();

  /* ── Mobile nav + dropdowns ────────────────────────────────────────────── */
  (function initNav() {
    var burger = document.getElementById("navBurger");
    var links = document.getElementById("navLinks");
    if (!burger || !links) return;

    function setMenu(open) {
      links.classList.toggle("open", open);
      burger.classList.toggle("open", open);
      burger.setAttribute("aria-expanded", String(open));
      // While the phone dropdown is open, lock the page behind it so only
      // the menu itself scrolls; the page scroll resumes when it closes.
      document.body.style.overflow = open ? "hidden" : "";
    }

    burger.addEventListener("click", function () {
      setMenu(!links.classList.contains("open"));
    });
    document.addEventListener("click", function (e) {
      if (!links.classList.contains("open")) return;
      if (!links.contains(e.target) && !burger.contains(e.target)) {
        setMenu(false);
      }
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && links.classList.contains("open")) setMenu(false);
    });
    // If the window grows past the phone breakpoint while the dropdown is
    // open (resize, tablet rotation), close it so the page scroll unlocks.
    window.addEventListener("resize", function () {
      if (links.classList.contains("open") && window.innerWidth > 900) setMenu(false);
    });

    // Dropdown toggles
    document.querySelectorAll(".nav-dropdown > .dropdown-toggle").forEach(function (toggle) {
      toggle.addEventListener("click", function (e) {
        e.preventDefault();
        var dd = toggle.closest(".nav-dropdown");
        var wasOpen = dd.classList.contains("open");
        // Close siblings
        document.querySelectorAll(".nav-dropdown.open").forEach(function (other) {
          other.classList.remove("open");
        });
        if (!wasOpen) dd.classList.add("open");
      });
    });
    document.addEventListener("click", function (e) {
      if (!e.target.closest(".nav-dropdown")) {
        document.querySelectorAll(".nav-dropdown.open").forEach(function (dd) {
          dd.classList.remove("open");
        });
      }
    });

  })();

  /* ── Admin console: profile avatar menu (Log out) ────────────────────── */
  (function initAdminUserMenu() {
    var wrap = document.querySelector(".admin-user");
    var toggle = wrap && wrap.querySelector(".admin-user-toggle");
    if (!wrap || !toggle) return;
    toggle.addEventListener("click", function (e) {
      e.stopPropagation();
      var open = wrap.classList.toggle("open");
      toggle.setAttribute("aria-expanded", String(open));
    });
    document.addEventListener("click", function (e) {
      if (wrap.classList.contains("open") && !e.target.closest(".admin-user")) {
        wrap.classList.remove("open");
        toggle.setAttribute("aria-expanded", "false");
      }
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && wrap.classList.contains("open")) {
        wrap.classList.remove("open");
        toggle.setAttribute("aria-expanded", "false");
      }
    });
  })();

  /* ── Scroll reveal ─────────────────────────────────────────────────────── */
  (function initReveal() {
    var els = document.querySelectorAll(".reveal");
    if (!els.length) return;
    if (!("IntersectionObserver" in window)) {
      els.forEach(function (el) { el.classList.add("visible"); });
      return;
    }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          io.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -40px 0px" });
    els.forEach(function (el) { io.observe(el); });
  })();

  /* ── Animated counters ─────────────────────────────────────────────────── */
  (function initCounters() {
    var els = document.querySelectorAll("[data-count]");
    if (!els.length) return;
    function animate(el) {
      var target = parseFloat(el.getAttribute("data-count"));
      var decimals = parseInt(el.getAttribute("data-decimals") || "0", 10);
      var duration = 1400;
      var start = null;
      function step(ts) {
        if (!start) start = ts;
        var p = Math.min((ts - start) / duration, 1);
        var eased = 1 - Math.pow(1 - p, 3);
        el.textContent = (target * eased).toFixed(decimals);
        if (p < 1) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    }
    if ("IntersectionObserver" in window) {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            animate(entry.target);
            io.unobserve(entry.target);
          }
        });
      }, { threshold: 0.4 });
      els.forEach(function (el) { io.observe(el); });
    } else {
      els.forEach(animate);
    }
  })();

  /* ── Tabs ──────────────────────────────────────────────────────────────── */
  (function initTabs() {
    document.querySelectorAll("[data-tabs]").forEach(function (group) {
      var btns = group.querySelectorAll(".tab-btn");
      btns.forEach(function (btn) {
        btn.addEventListener("click", function () {
          var targetId = btn.getAttribute("data-tab");
          btns.forEach(function (b) { b.classList.remove("active"); });
          btn.classList.add("active");
          group.querySelectorAll(".tab-panel").forEach(function (panel) {
            panel.classList.toggle("active", panel.id === targetId);
          });
        });
      });
    });
  })();

  /* ── FAQ accordion ─────────────────────────────────────────────────────── */
  (function initFaq() {
    document.querySelectorAll(".faq-question").forEach(function (q) {
      q.addEventListener("click", function () {
        var item = q.closest(".faq-item");
        var answer = item.querySelector(".faq-answer");
        var isOpen = item.classList.contains("open");
        // Close others
        document.querySelectorAll(".faq-item.open").forEach(function (o) {
          if (o !== item) {
            o.classList.remove("open");
            o.querySelector(".faq-answer").style.maxHeight = "0px";
          }
        });
        item.classList.toggle("open", !isOpen);
        answer.style.maxHeight = isOpen ? "0px" : answer.scrollHeight + "px";
      });
    });
  })();

  /* ── Toasts ────────────────────────────────────────────────────────────── */
  var toastRegion = document.getElementById("toastRegion");
  function toast(message, type) {
    if (!toastRegion) return;
    var t = document.createElement("div");
    t.className = "toast toast-" + (type || "info");
    var icon = type === "success"
      ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="20 6 9 17 4 12"/></svg>'
      : type === "error"
        ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>'
        : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>';
    // Icon is a static SVG string (safe); the message text is inserted with
    // textContent so user-controlled flash content can never run as HTML.
    var iconWrap = document.createElement("span");
    iconWrap.innerHTML = icon;
    var textWrap = document.createElement("span");
    textWrap.textContent = message;
    t.appendChild(iconWrap);
    t.appendChild(textWrap);
    toastRegion.appendChild(t);
    setTimeout(function () {
      t.classList.add("out");
      setTimeout(function () { t.remove(); }, 250);
    }, 3800);
  }
  window.SCP = { toast: toast };

  /* ── Flashed messages → toasts (server-rendered auth feedback) ─────────── */
  (function initFlash() {
    var el = document.getElementById("flashData");
    if (!el) return;
    try {
      var messages = JSON.parse(el.textContent);
      (messages || []).forEach(function (pair, i) {
        var type = pair[0] === "message" ? "info" : pair[0];
        var msg = pair[1];
        setTimeout(function () { toast(msg, type); }, 300 + i * 250);
      });
    } catch (e) { /* ignore malformed flash data */ }
  })();

  /* ── Modal ─────────────────────────────────────────────────────────────── */
  (function initModal() {
    var backdrop = document.getElementById("modalRoot");
    if (!backdrop) return;
    document.querySelectorAll("[data-modal-open]").forEach(function (trigger) {
      trigger.addEventListener("click", function (e) {
        e.preventDefault();
        var id = trigger.getAttribute("data-modal-open");
        var target = document.getElementById(id);
        if (!target) return;
        backdrop.appendChild(target);
        target.classList.add("open");
        backdrop.classList.add("open");
        document.body.style.overflow = "hidden";
        var close = target.querySelector("[data-modal-close]");
        if (close) close.focus();
      });
    });
    backdrop.addEventListener("click", function (e) {
      if (e.target === backdrop) closeModal();
    });
    document.querySelectorAll("[data-modal-close]").forEach(function (c) {
      c.addEventListener("click", closeModal);
    });
    function closeModal() {
      backdrop.classList.remove("open");
      document.querySelectorAll(".modal.open").forEach(function (m) {
        m.classList.remove("open");
      });
      document.body.style.overflow = "";
    }
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && backdrop.classList.contains("open")) closeModal();
    });
  })();

  /* ── Back to top ───────────────────────────────────────────────────────── */
  (function initBackToTop() {
    var btn = document.getElementById("backToTop");
    if (!btn) return;
    window.addEventListener("scroll", function () {
      btn.classList.toggle("show", window.scrollY > 600);
    }, { passive: true });
    btn.addEventListener("click", function () {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  })();

  /* ── Copy-to-clipboard on code blocks ──────────────────────────────────── */
  (function initCopy() {
    document.querySelectorAll(".code-block").forEach(function (block) {
      var btn = document.createElement("button");
      btn.className = "copy-btn";
      btn.type = "button";
      btn.setAttribute("aria-label", "Copy code");
      btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
      btn.addEventListener("click", function () {
        var text = block.querySelector("pre").innerText;
        if (navigator.clipboard) {
          navigator.clipboard.writeText(text).then(function () {
            toast("Copied to clipboard", "success");
          });
        } else {
          toast("Copy not supported in this browser", "error");
        }
      });
      block.appendChild(btn);
    });
  })();

  /* ── Pricing billing toggle ────────────────────────────────────────────── */
  (function initPricing() {
    var toggle = document.getElementById("billingToggle");
    if (!toggle) return;
    toggle.addEventListener("change", function () {
      var yearly = toggle.checked;
      document.querySelectorAll("[data-monthly]").forEach(function (el) {
        el.textContent = yearly ? el.getAttribute("data-yearly") : el.getAttribute("data-monthly");
      });
      document.querySelectorAll(".price-period").forEach(function (el) {
        el.textContent = yearly ? "/year, billed annually" : "/month";
      });
    });
  })();

  /* ── Form validation (client-side) ─────────────────────────────────────── */
  (function initValidation() {
    document.querySelectorAll("form[data-validate]").forEach(function (form) {
      form.addEventListener("submit", function (e) {
        var valid = true;
        var emptyFields = false;
        var badEmail = false;
        var boxUnchecked = false;
        form.querySelectorAll("[required]").forEach(function (input) {
          var field = input.closest(".field") || input.parentElement;
          var empty = false;
          if (input.type === "checkbox") {
            empty = !input.checked;
            if (empty) boxUnchecked = true;
          } else {
            empty = !input.value.trim();
            if (empty) emptyFields = true;
          }
          if (input.type === "email" && input.value && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(input.value)) {
            field.classList.add("has-error");
            valid = false;
            badEmail = true;
          } else if (empty) {
            field.classList.add("has-error");
            valid = false;
          } else {
            field.classList.remove("has-error");
          }
        });
        if (!valid) {
          e.preventDefault();
          // Specific, actionable messages instead of a generic "fix the fields".
          var msg = emptyFields
            ? "Please fill the required fields"
            : badEmail
              ? "Please enter a valid email address"
              : "Please tick the box to continue";
          toast(msg, "error");
        }
      });
      form.querySelectorAll(".input, .select, .textarea").forEach(function (input) {
        input.addEventListener("input", function () {
          var field = input.closest(".field") || input.parentElement;
          if (input.value.trim()) field.classList.remove("has-error");
        });
      });
      form.querySelectorAll("input[type=checkbox]").forEach(function (input) {
        input.addEventListener("change", function () {
          var field = input.closest(".field") || input.parentElement;
          if (input.checked) field.classList.remove("has-error");
        });
      });
    });
  })();

  /* ── Same-city guard for origin/destination selects ────────────────────── */
  (function initSameCityGuard() {
    function sync(form) {
      var origin = form.querySelector('[name="origin"]');
      var dest = form.querySelector('[name="destination"]');
      if (!origin || !dest) return;
      var o = (origin.value || "").toLowerCase();
      var d = (dest.value || "").toLowerCase();

      // Disable the currently chosen origin in the destination dropdown and
      // vice versa, so the same city can never be picked for both.
      Array.prototype.forEach.call(dest.options, function (opt) {
        opt.disabled = opt.value && opt.value.toLowerCase() === o;
      });
      Array.prototype.forEach.call(origin.options, function (opt) {
        opt.disabled = opt.value && opt.value.toLowerCase() === d;
      });

      // If the values collided (e.g. after a change), reset the later one.
      if (o && o === d) {
        if (form._lastChanged === "origin") {
          dest.value = "";
        } else {
          origin.value = "";
        }
      }
    }

    document.querySelectorAll("form[data-validate]").forEach(function (form) {
      var origin = form.querySelector('[name="origin"]');
      var dest = form.querySelector('[name="destination"]');
      if (!origin || !dest) return;
      form._lastChanged = "dest";
      origin.addEventListener("change", function () {
        form._lastChanged = "origin";
        sync(form);
      });
      dest.addEventListener("change", function () {
        form._lastChanged = "dest";
        sync(form);
      });
      sync(form);
    });
  })();

  /* ── Demo-only forms: show a toast instead of submitting ───────────────── */
  (function initDemoForms() {
    document.querySelectorAll("form[data-demo]").forEach(function (form) {
    form.addEventListener("submit", function (e) {
      e.preventDefault();
      // The shared validation handler already flags invalid fields and toasts;
      // bail out here so we never double-toast on invalid input. Note: checkbox
      // errors live on their <label> parent, so check for any .has-error.
      var invalid = form.querySelectorAll(".has-error").length > 0;
      if (invalid) return;
      var action = form.getAttribute("data-demo");
      toast("Submitted — this is a preview build, nothing was sent", "success");
      if (action === "reset") form.reset();
    });
    });
  })();

  /* ── Account page: show / hide / copy API key ─────────────────────────── */
  (function initAccountKey() {
    var keyEl = document.getElementById("apiKeyValue");
    var showBtn = document.getElementById("keyShowBtn");
    var hideBtn = document.getElementById("keyHideBtn");
    var copyBtn = document.getElementById("keyCopyBtn");
    if (!keyEl || !showBtn) return;

    var realKey = null;

    function setMasked() {
      realKey = null;
      keyEl.textContent = keyEl.getAttribute("data-masked") || "No key generated yet";
      showBtn.style.display = "";
      hideBtn.style.display = "none";
      copyBtn.disabled = true;
    }

    function setRevealed(key) {
      realKey = key;
      keyEl.textContent = key;
      showBtn.style.display = "none";
      hideBtn.style.display = "";
      copyBtn.disabled = false;
    }

    showBtn.addEventListener("click", function () {
      fetch("/account/key", { headers: { "Accept": "application/json" } })
        .then(function (resp) {
          return resp.json().then(function (data) {
            if (!resp.ok || !data.api_key) {
              toast((data && data.error) || "No key stored yet — regenerate one", "info");
              return null;
            }
            return data.api_key;
          });
        })
        .then(function (key) {
          if (key) setRevealed(key);
        })
        .catch(function () { toast("Could not load your API key", "error"); });
    });

    if (hideBtn) hideBtn.addEventListener("click", setMasked);

    if (copyBtn) copyBtn.addEventListener("click", function () {
      var text = realKey || keyEl.textContent;
      if (!text || text.indexOf("…") !== -1 || text.indexOf("••") !== -1) {
        toast("Reveal the key first with Show, then copy it", "info");
        return;
      }
      function done() { toast("API key copied to clipboard", "success"); }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done).catch(function () {
          fallbackCopy(text, done);
        });
      } else {
        fallbackCopy(text, done);
      }
    });

    function fallbackCopy(text, done) {
      var ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      try {
        document.execCommand("copy");
        done();
      } catch (e) {
        toast("Copy not supported in this browser", "error");
      }
      document.body.removeChild(ta);
    }

    setMasked();
  })();

  /* ── Password strength (text only) ─────────────────────────────────────── */
  (function initPasswordStrength() {
    var input = document.getElementById("registerPassword");
    var label = document.getElementById("strengthLabel");
    if (!input || !label) return;
    var DEFAULT_TEXT = "At least 8 characters";
    input.addEventListener("input", function () {
      var v = input.value;
      if (!v) {
        label.textContent = DEFAULT_TEXT;
        label.className = "strength-label";
        return;
      }
      var score = 0;
      if (v.length >= 8) score++;
      if (/[A-Z]/.test(v) && /[a-z]/.test(v)) score++;
      if (/\d/.test(v)) score++;
      if (/[^A-Za-z0-9]/.test(v)) score++;
      var names = ["Too weak", "Weak", "Fair", "Good", "Strong"];
      var classes = ["is-too-weak", "is-weak", "is-fair", "is-good", "is-strong"];
      label.textContent = "Password: " + names[score];
      label.className = "strength-label " + classes[score];
    });
  })();



  /* ── Table filtering (admin) ───────────────────────────────────────────── */
  (function initTableFilter() {
    var search = document.getElementById("tableSearch");
    if (!search) return;
    search.addEventListener("input", function () {
      var q = search.value.trim().toLowerCase();
      document.querySelectorAll("table tbody tr[data-row]").forEach(function (row) {
        row.style.display = row.textContent.toLowerCase().indexOf(q) !== -1 ? "" : "none";
      });
    });
  })();

  /* ── Prediction form live preview (index / demo) ───────────────────────── */
  function bindPredictionPreview() {
    var form = document.getElementById("predictForm");
    if (!form) return;
    var btn = document.getElementById("predictBtn");
    var btnText = document.getElementById("btnText");

    form.addEventListener("submit", function (e) {
      // Only show the loading state when the form is actually valid — the
      // validation handler owns invalid submissions and prevents the POST.
      if (!form.checkValidity()) return;
      var overlay = document.getElementById("predictOverlay");
      if (overlay) {
        e.preventDefault();
        overlay.classList.add("open");
        overlay.setAttribute("aria-hidden", "false");
        document.body.style.overflow = "hidden";
      }
      btn.disabled = true;
      btn.classList.add("is-loading");
      btn.innerHTML = '<span class="spinner" aria-hidden="true"></span><span id="btnText">Calculating…</span>';
      // Give the loading animation time to play before the POST navigates.
      if (overlay) {
        setTimeout(function () { form.submit(); }, 1600);
      }
    });

    var map = {
      origin: "previewOrigin", destination: "previewDest", mode: "previewMode",
      nature_of_consignment: "previewType", booking_weekday: "previewDay"
    };
    function selectedText(sel) {
      return sel.options[sel.selectedIndex] ? sel.options[sel.selectedIndex].text : "—";
    }
    function fmt(el, suffix) {
      var v = parseFloat(el.value);
      return (isNaN(v) ? "0" : v.toFixed(2)) + suffix;
    }
    function update() {
      Object.keys(map).forEach(function (id) {
        var el = document.getElementById(id);
        var out = document.getElementById(map[id]);
        if (el && out) out.textContent = selectedText(el);
      });
      var pieces = document.getElementById("previewPieces");
      var total = document.getElementById("total_pieces");
      if (pieces && total) pieces.textContent = total.value || "0";
      var act = document.getElementById("previewActWt");
      var vol = document.getElementById("previewVolWt");
      var chg = document.getElementById("previewChgWt");
      var a = document.getElementById("actual_weight");
      var v = document.getElementById("volumetric_weight");
      var c = document.getElementById("chargeable_weight");
      if (act && a) act.textContent = fmt(a, " kg");
      if (vol && v) vol.textContent = fmt(v, " kg");
      if (chg && c) chg.textContent = fmt(c, " kg");
    }
    form.querySelectorAll("select, input").forEach(function (el) {
      el.addEventListener("change", update);
      el.addEventListener("input", update);
    });
    update();
  }
  bindPredictionPreview();



  /* ── Report download + demo utility buttons ───────────────────────────── */
  (function initUtilityButtons() {
    document.querySelectorAll("[data-print-report]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        toast("Generating PDF report…", "info");
        setTimeout(function () { window.print(); }, 350);
      });
    });
  })();

  /* ── Current year ──────────────────────────────────────────────────────── */
  document.querySelectorAll("[data-year]").forEach(function (el) {
    el.textContent = String(new Date().getFullYear());
  });
})();
