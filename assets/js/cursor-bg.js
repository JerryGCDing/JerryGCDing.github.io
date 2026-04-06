/**
 * cursor-bg.js — subtle sci-fi dot-grid cursor tracking background animation.
 *
 * To disable: set `enable_cursor_animation: false` in _config.yml.
 * To remove entirely: delete this file and the conditional block in
 * _includes/scripts/misc.liquid.
 *
 * No external dependencies. Self-contained IIFE. Nothing added to global scope.
 */
(function () {
  'use strict';

  // Bail out for users who prefer reduced motion.
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  // ---------------------------------------------------------------------------
  // Config — tweak here to adjust the feel without touching the draw loop.
  // ---------------------------------------------------------------------------
  var CFG = {
    spacing:      80,    // px between grid points (larger = sparser)
    attractR:     200,   // px — cursor influence radius
    attractK:     0.14,  // max fractional displacement toward cursor (0–1)
    lerpSpeed:    0.10,  // how fast points chase their target per frame
    lineDist:     116,   // px — max actual distance to draw a connecting line
    dotR:         1.3,   // dot radius in px
    // Colors — intentionally muted/desaturated to stay ambient
    //   light mode: slate-blue at very low opacity
    //   dark mode:  steel-cyan at very low opacity
    light: { r: 110, g: 130, b: 175 },
    dark:  { r:  70, g: 155, b: 195 },
    // Opacity levels
    dotBaseAlpha:  0.22,  // dot at rest
    dotNearAlpha:  0.55,  // dot when cursor is close
    dotNearR:      160,   // px — distance at which dot reaches dotNearAlpha
    lineBaseAlpha: 0.13,  // grid lines at rest
    lineNearAlpha: 0.28,  // grid lines boosted near cursor
    glowAlpha:     0.07,  // radial glow at cursor center
    glowR:         110,   // px — glow radius
  };

  // ---------------------------------------------------------------------------
  // Canvas setup
  // ---------------------------------------------------------------------------
  var canvas = document.createElement('canvas');
  canvas.setAttribute('aria-hidden', 'true');
  canvas.style.cssText = [
    'position:fixed',
    'top:0',
    'left:0',
    'width:100%',
    'height:100%',
    'z-index:-1',
    'pointer-events:none',
    'display:block',
  ].join(';');
  document.body.insertBefore(canvas, document.body.firstChild);

  var ctx = canvas.getContext('2d');
  var W = 0, H = 0, cols = 0, rows = 0;
  var points = [];
  var mouse = { x: -99999, y: -99999 };

  // ---------------------------------------------------------------------------
  // Grid
  // ---------------------------------------------------------------------------
  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
    cols = Math.ceil(W / CFG.spacing) + 2;
    rows = Math.ceil(H / CFG.spacing) + 2;
    buildGrid();
  }

  function buildGrid() {
    points = new Array(cols * rows);
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        var ox = c * CFG.spacing;
        var oy = r * CFG.spacing;
        points[r * cols + c] = { ox: ox, oy: oy, x: ox, y: oy };
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Theme color
  // ---------------------------------------------------------------------------
  function rgb() {
    var dark = document.documentElement.getAttribute('data-theme') === 'dark';
    return dark ? CFG.dark : CFG.light;
  }

  // ---------------------------------------------------------------------------
  // Draw loop
  // ---------------------------------------------------------------------------
  function tick() {
    requestAnimationFrame(tick);
    ctx.clearRect(0, 0, W, H);

    var c = rgb();
    var mx = mouse.x, my = mouse.y;
    var attractR2 = CFG.attractR * CFG.attractR;

    // --- update positions ---
    for (var i = 0; i < points.length; i++) {
      var p = points[i];
      var dx = mx - p.ox, dy = my - p.oy;
      var d2 = dx * dx + dy * dy;
      var tx, ty;
      if (d2 < attractR2) {
        var d  = Math.sqrt(d2);
        var k  = (1 - d / CFG.attractR) * CFG.attractK;
        tx = p.ox + dx * k;
        ty = p.oy + dy * k;
      } else {
        tx = p.ox;
        ty = p.oy;
      }
      p.x += (tx - p.x) * CFG.lerpSpeed;
      p.y += (ty - p.y) * CFG.lerpSpeed;
    }

    // --- draw grid edges (O(n) — only right and bottom neighbors) ---
    for (var i = 0; i < points.length; i++) {
      var a = points[i];
      var col = i % cols;

      // right neighbor
      if (col + 1 < cols) {
        var b = points[i + 1];
        drawEdge(a, b, c, mx, my);
      }
      // bottom neighbor
      if (i + cols < points.length) {
        var b = points[i + cols];
        drawEdge(a, b, c, mx, my);
      }
    }

    // --- draw dots ---
    for (var i = 0; i < points.length; i++) {
      var p  = points[i];
      var dx = mx - p.x, dy = my - p.y;
      var d  = Math.sqrt(dx * dx + dy * dy);
      var alpha;
      if (d < CFG.dotNearR) {
        alpha = CFG.dotBaseAlpha + (CFG.dotNearAlpha - CFG.dotBaseAlpha) * (1 - d / CFG.dotNearR);
      } else {
        alpha = CFG.dotBaseAlpha;
      }
      ctx.beginPath();
      ctx.arc(p.x, p.y, CFG.dotR, 0, 6.2832);
      ctx.fillStyle = 'rgba(' + c.r + ',' + c.g + ',' + c.b + ',' + alpha.toFixed(3) + ')';
      ctx.fill();
    }

    // --- cursor glow ---
    if (mx > -9999) {
      var grad = ctx.createRadialGradient(mx, my, 0, mx, my, CFG.glowR);
      var gBase = 'rgba(' + c.r + ',' + c.g + ',' + c.b + ',';
      grad.addColorStop(0,    gBase + CFG.glowAlpha + ')');
      grad.addColorStop(0.45, gBase + (CFG.glowAlpha * 0.4).toFixed(3) + ')');
      grad.addColorStop(1,    gBase + '0)');
      ctx.beginPath();
      ctx.arc(mx, my, CFG.glowR, 0, 6.2832);
      ctx.fillStyle = grad;
      ctx.fill();
    }
  }

  function drawEdge(a, b, c, mx, my) {
    var dx = a.x - b.x, dy = a.y - b.y;
    var d  = Math.sqrt(dx * dx + dy * dy);
    if (d > CFG.lineDist) return;

    // Boost opacity if either endpoint is near cursor
    var dac = dist2pt(mx, my, a.x, a.y);
    var dbc = dist2pt(mx, my, b.x, b.y);
    var near = Math.min(dac, dbc);
    var alpha;
    if (near < CFG.dotNearR) {
      alpha = CFG.lineBaseAlpha + (CFG.lineNearAlpha - CFG.lineBaseAlpha) * (1 - near / CFG.dotNearR);
    } else {
      alpha = CFG.lineBaseAlpha;
    }
    // Also fade by how stretched the edge is
    alpha *= (1 - d / CFG.lineDist) * 1.4;
    alpha = Math.min(alpha, CFG.lineNearAlpha);

    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.strokeStyle = 'rgba(' + c.r + ',' + c.g + ',' + c.b + ',' + alpha.toFixed(3) + ')';
    ctx.lineWidth = 0.5;
    ctx.stroke();
  }

  function dist2pt(x1, y1, x2, y2) {
    var dx = x1 - x2, dy = y1 - y2;
    return Math.sqrt(dx * dx + dy * dy);
  }

  // ---------------------------------------------------------------------------
  // Events
  // ---------------------------------------------------------------------------
  document.addEventListener('mousemove', function (e) {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
  });

  document.addEventListener('mouseleave', function () {
    mouse.x = -99999;
    mouse.y = -99999;
  });

  window.addEventListener('resize', resize);

  // ---------------------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------------------
  resize();
  tick();
})();
