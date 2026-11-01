// LLM Systems from Scratch: theme toggle, navigation, scroll-spy and search.
(function () {
  "use strict";
  var root = document.documentElement;

  // ------------------------------------------------------------- theme
  function setTheme(t) {
    root.setAttribute("data-theme", t);
    try { localStorage.setItem("theme", t); } catch (e) {}
  }
  var tbtn = document.getElementById("theme-toggle");
  if (tbtn) tbtn.addEventListener("click", function () {
    setTheme(root.getAttribute("data-theme") === "dark" ? "light" : "dark");
  });

  // ------------------------------------------------------------- sidebar
  var page = location.pathname.split("/").pop() || "index.html";
  document.querySelectorAll(".chapters a").forEach(function (a) {
    if (a.getAttribute("data-page") === page) {
      a.classList.add("active");
      var sb = document.getElementById("sidebar");
      if (sb) sb.scrollTop = Math.max(0, a.offsetTop - sb.clientHeight / 3);
    }
  });
  var nbtn = document.getElementById("nav-toggle");
  if (nbtn) nbtn.addEventListener("click", function () { document.body.classList.toggle("nav-open"); });
  document.addEventListener("click", function (e) {
    if (!document.body.classList.contains("nav-open")) return;
    if (e.target.closest("#sidebar") || e.target.closest("#nav-toggle")) return;
    document.body.classList.remove("nav-open");
  });

  // ------------------------------------------------------------- scroll-spy for "On this page"
  var tocLinks = Array.prototype.slice.call(document.querySelectorAll(".pagetoc a"));
  if (tocLinks.length && "IntersectionObserver" in window) {
    var map = {};
    tocLinks.forEach(function (a) { map[decodeURIComponent(a.getAttribute("href").slice(1))] = a; });
    var current = null;
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) {
          var a = map[en.target.id];
          if (a) {
            if (current) current.classList.remove("active");
            a.classList.add("active"); current = a;
          }
        }
      });
    }, { rootMargin: "-70px 0px -70% 0px" });
    Object.keys(map).forEach(function (id) {
      var el = document.getElementById(id);
      if (el) obs.observe(el);
    });
  }

  // ------------------------------------------------------------- search
  var overlay = document.getElementById("search-overlay");
  var input = document.getElementById("search-input");
  var results = document.getElementById("search-results");
  var index = null, sel = -1;

  function openSearch() {
    if (!overlay) return;
    overlay.hidden = false; input.value = ""; results.innerHTML = ""; input.focus();
    if (!index) {
      fetch("search.json").then(function (r) { return r.json(); })
        .then(function (d) { index = d; run(); })
        .catch(function () { results.innerHTML = "<li>Search index unavailable.</li>"; });
    }
  }
  function closeSearch() { if (overlay) overlay.hidden = true; }
  function esc(s) { return s.replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function highlight(text, terms) {
    var out = esc(text);
    terms.forEach(function (t) {
      if (!t) return;
      var re = new RegExp("(" + t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "ig");
      out = out.replace(re, "<mark>$1</mark>");
    });
    return out;
  }
  function run() {
    if (!index) return;
    var q = input.value.trim().toLowerCase();
    if (q.length < 2) { results.innerHTML = ""; return; }
    var terms = q.split(/\s+/);
    var scored = [];
    for (var i = 0; i < index.length; i++) {
      var e = index[i], t = e.t.toLowerCase(), x = e.x.toLowerCase(), s = 0, ok = true;
      for (var j = 0; j < terms.length; j++) {
        var inT = t.indexOf(terms[j]) >= 0, inX = x.indexOf(terms[j]) >= 0;
        if (!inT && !inX) { ok = false; break; }
        s += (inT ? 10 : 0) + (inX ? 1 : 0);
      }
      if (ok) scored.push([s, e]);
    }
    scored.sort(function (a, b) { return b[0] - a[0]; });
    results.innerHTML = scored.slice(0, 30).map(function (p) {
      var e = p[1], x = e.x, k = x.toLowerCase().indexOf(terms[0]);
      var snip = x.slice(Math.max(0, k - 60), Math.max(0, k - 60) + 180);
      return '<li><a href="' + e.u + '"><div class="r-page">' + esc(e.p) + '</div><div class="r-title">' +
        highlight(e.t, terms) + '</div><div class="r-snip">' + highlight(snip, terms) + "</div></a></li>";
    }).join("") || "<li style='padding:12px'>No results.</li>";
    sel = -1;
  }
  if (input) {
    input.addEventListener("input", run);
    input.addEventListener("keydown", function (e) {
      var links = results.querySelectorAll("a");
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        sel = Math.max(0, Math.min(links.length - 1, sel + (e.key === "ArrowDown" ? 1 : -1)));
        links.forEach(function (a, i) { a.classList.toggle("sel", i === sel); });
        if (links[sel]) links[sel].scrollIntoView({ block: "nearest" });
      } else if (e.key === "Enter" && links.length) {
        (links[sel] || links[0]).click();
      }
    });
  }
  var sbtn = document.getElementById("search-open");
  if (sbtn) sbtn.addEventListener("click", openSearch);
  if (overlay) overlay.addEventListener("click", function (e) { if (e.target === overlay) closeSearch(); });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") closeSearch();
    var tag = (e.target.tagName || "").toLowerCase();
    if (e.key === "/" && tag !== "input" && tag !== "textarea") { e.preventDefault(); openSearch(); }
  });
})();
