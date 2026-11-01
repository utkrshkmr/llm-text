// LLM Systems from Scratch: reader controls, collapsible panels, scroll-spy, figure zoom, search.
(function () {
  "use strict";
  var root = document.documentElement;
  function load(k, d) { try { var v = localStorage.getItem(k); return v === null ? d : v; } catch (e) { return d; } }
  function save(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }

  // ------------------------------------------------------------- theme (light -> sepia -> dark)
  var THEMES = ["light", "sepia", "dark"];
  var themeBtn = document.getElementById("theme-toggle");
  var themeLabel = document.getElementById("theme-label");
  function showTheme() { if (themeLabel) themeLabel.textContent = root.getAttribute("data-theme"); }
  showTheme();
  if (themeBtn) themeBtn.addEventListener("click", function () {
    var i = THEMES.indexOf(root.getAttribute("data-theme"));
    var t = THEMES[(i + 1) % THEMES.length];
    root.setAttribute("data-theme", t); save("theme", t); showTheme();
  });

  // ------------------------------------------------------------- font size and family
  function setFs(px) {
    px = Math.max(15, Math.min(25, px));
    root.style.setProperty("--fs", px + "px"); save("fs", String(px));
  }
  var fsDown = document.getElementById("fs-down"), fsUp = document.getElementById("fs-up");
  function curFs() { return parseFloat(getComputedStyle(root).getPropertyValue("--fs")) || 19; }
  if (fsDown) fsDown.addEventListener("click", function () { setFs(curFs() - 1); });
  if (fsUp) fsUp.addEventListener("click", function () { setFs(curFs() + 1); });
  var fontBtn = document.getElementById("font-toggle");
  if (fontBtn) {
    fontBtn.setAttribute("aria-pressed", root.getAttribute("data-font") === "sans" ? "true" : "false");
    fontBtn.addEventListener("click", function () {
      var f = root.getAttribute("data-font") === "sans" ? "serif" : "sans";
      root.setAttribute("data-font", f); save("font", f);
      fontBtn.setAttribute("aria-pressed", f === "sans" ? "true" : "false");
    });
  }

  // ------------------------------------------------------------- collapsible side panels
  var mobile = function () { return window.matchMedia("(max-width: 920px)").matches; };
  var navBtn = document.getElementById("nav-toggle");
  if (navBtn) navBtn.addEventListener("click", function () {
    if (mobile()) { document.body.classList.toggle("nav-open"); return; }
    var v = root.getAttribute("data-nav") === "closed" ? "open" : "closed";
    root.setAttribute("data-nav", v); save("nav", v);
  });
  var tocBtn = document.getElementById("toc-toggle");
  if (tocBtn) tocBtn.addEventListener("click", function () {
    var v = root.getAttribute("data-toc") === "closed" ? "open" : "closed";
    root.setAttribute("data-toc", v); save("toc", v);
  });
  document.addEventListener("click", function (e) {
    if (!document.body.classList.contains("nav-open")) return;
    if (e.target.closest("#sidebar") || e.target.closest("#nav-toggle")) return;
    document.body.classList.remove("nav-open");
  });

  // ------------------------------------------------------------- chapter list: collapsible parts
  var page = location.pathname.split("/").pop() || "index.html";
  var openParts = {};
  try { openParts = JSON.parse(load("openParts", "{}")) || {}; } catch (e) { openParts = {}; }
  document.querySelectorAll(".nav-part").forEach(function (p) {
    var id = p.getAttribute("data-part");
    var hasActive = !!p.querySelector('a[data-page="' + page + '"]');
    if (hasActive || openParts[id]) p.classList.add("open");
    var b = p.querySelector(".part-toggle");
    if (b) b.addEventListener("click", function () {
      p.classList.toggle("open");
      openParts[id] = p.classList.contains("open");
      save("openParts", JSON.stringify(openParts));
    });
  });
  document.querySelectorAll(".sidebar a").forEach(function (a) {
    if (a.getAttribute("data-page") === page) {
      a.classList.add("active");
      var sb = document.getElementById("sidebar");
      if (sb) setTimeout(function () { sb.scrollTop = Math.max(0, a.offsetTop - sb.clientHeight / 3); }, 0);
    }
  });

  // ------------------------------------------------------------- "On this page": collapsible sub-sections
  var toc = document.querySelector(".pagetoc nav");
  if (toc) {
    toc.querySelectorAll("li").forEach(function (li) {
      var sub = li.querySelector(":scope > ul");
      if (!sub) return;
      var t = document.createElement("button");
      t.className = "sub-toggle"; t.setAttribute("aria-label", "Expand section"); t.textContent = "▸";
      t.addEventListener("click", function (e) { e.preventDefault(); li.classList.toggle("open"); });
      li.insertBefore(t, li.firstChild);
    });
  }
  var tocLinks = Array.prototype.slice.call(document.querySelectorAll(".pagetoc a"));
  if (tocLinks.length && "IntersectionObserver" in window) {
    var map = {};
    tocLinks.forEach(function (a) { map[decodeURIComponent(a.getAttribute("href").slice(1))] = a; });
    var current = null;
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (!en.isIntersecting) return;
        var a = map[en.target.id];
        if (!a) return;
        if (current) current.classList.remove("active");
        a.classList.add("active"); current = a;
        var li = a.closest("li");
        while (li) { if (li.querySelector(":scope > ul")) li.classList.add("open"); li = li.parentElement.closest("li"); }
      });
    }, { rootMargin: "-80px 0px -70% 0px" });
    Object.keys(map).forEach(function (id) { var el = document.getElementById(id); if (el) obs.observe(el); });
  }

  // ------------------------------------------------------------- figure zoom
  var zoom = document.getElementById("zoom-overlay");
  if (zoom) {
    var zimg = zoom.querySelector("img");
    document.querySelectorAll("img.tikz").forEach(function (img) {
      img.addEventListener("click", function () { zimg.src = img.src; zoom.hidden = false; });
    });
    zoom.addEventListener("click", function () { zoom.hidden = true; zimg.removeAttribute("src"); });
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
        .catch(function () { results.innerHTML = "<li style='padding:12px'>Search index unavailable.</li>"; });
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
    var terms = q.split(/\s+/), scored = [];
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
      } else if (e.key === "Enter" && links.length) { (links[sel] || links[0]).click(); }
    });
  }
  var sbtn = document.getElementById("search-open");
  if (sbtn) sbtn.addEventListener("click", openSearch);
  if (overlay) overlay.addEventListener("click", function (e) { if (e.target === overlay) closeSearch(); });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") { closeSearch(); if (zoom) zoom.hidden = true; }
    var tag = (e.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea") return;
    if (e.key === "/") { e.preventDefault(); openSearch(); }
    if (e.key === "[" && navBtn) navBtn.click();
    if (e.key === "]" && tocBtn) tocBtn.click();
  });
})();
