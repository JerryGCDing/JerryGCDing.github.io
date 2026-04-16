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

  // ---------------------------------------------------------------------------
  // Config — tweak here to adjust the feel without touching the draw loop.
  // ---------------------------------------------------------------------------
  var CFG = {
    spacing:      65,    // px between grid points
    attractR:     260,   // px — cursor influence radius
    attractK:     0.24,  // max fractional displacement toward cursor (0–1)
    lerpSpeed:    0.13,  // how fast points chase their target per frame
    lineDist:     100,   // px — max actual distance to draw a connecting line
    dotR:         2.0,   // dot radius in px
    // Colors — sci-fi palette
    //   light mode: electric blue
    //   dark mode:  electric cyan
    light: { r: 65,  g: 100, b: 210 },
    dark:  { r:  0,  g: 190, b: 255 },
    // Opacity levels
    dotBaseAlpha:  0.32,  // dot at rest
    dotNearAlpha:  0.85,  // dot when cursor is close
    dotNearR:      210,   // px — distance at which dot reaches dotNearAlpha
    lineBaseAlpha: 0.20,  // grid lines at rest
    lineNearAlpha: 0.55,  // grid lines boosted near cursor
    glowAlpha:     0.12,  // radial glow at cursor center
    glowR:         150,   // px — glow radius
    // Reticle — sci-fi corner-bracket cursor marker
    reticleSize:   18,    // half-width of the bracket square in px
    reticleTick:   7,     // arm length of each corner bracket
    reticleAlpha:  0.80,  // opacity of reticle
    // Ambient grid drift — animates even without cursor movement
    ambientAmp:   10,     // px — max drift displacement per point
    ambientFreqX: 0.0005, // x oscillation speed  (one cycle ≈ 12 s at 60 fps)
    ambientFreqY: 0.0004, // y oscillation speed — different to produce elliptical paths
    jitter:       18,     // px — max random offset applied to each point's origin at init
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
    var rect = canvas.getBoundingClientRect();
    W = canvas.width  = Math.round(rect.width);
    H = canvas.height = Math.round(rect.height);
    cols = Math.ceil(W / CFG.spacing) + 2;
    rows = Math.ceil(H / CFG.spacing) + 2;
    buildGrid();
  }

  function buildGrid() {
    points = new Array(cols * rows);
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        var ox = c * CFG.spacing + (Math.random() * 2 - 1) * CFG.jitter;
        var oy = r * CFG.spacing + (Math.random() * 2 - 1) * CFG.jitter;
        points[r * cols + c] = { ox: ox, oy: oy, x: ox, y: oy, phase: Math.random() * 6.2832 };
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
      var t   = performance.now();
      var ax  = CFG.ambientAmp * Math.sin(t * CFG.ambientFreqX + p.phase);
      var ay  = CFG.ambientAmp * Math.cos(t * CFG.ambientFreqY + p.phase);
      var tx, ty;
      if (d2 < attractR2) {
        var d  = Math.sqrt(d2);
        var k  = (1 - d / CFG.attractR) * CFG.attractK;
        tx = p.ox + ax + dx * k;
        ty = p.oy + ay + dy * k;
      } else {
        tx = p.ox + ax;
        ty = p.oy + ay;
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

      // --- cursor reticle ---
      drawReticle(mx, my, c);
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

  function drawReticle(mx, my, c) {
    var s = CFG.reticleSize;
    var t = CFG.reticleTick;
    var col = 'rgba(' + c.r + ',' + c.g + ',' + c.b + ',';

    // Outer ring
    ctx.beginPath();
    ctx.arc(mx, my, s + 10, 0, 6.2832);
    ctx.strokeStyle = col + (CFG.reticleAlpha * 0.3).toFixed(3) + ')';
    ctx.lineWidth = 0.7;
    ctx.stroke();

    // Corner brackets
    ctx.strokeStyle = col + CFG.reticleAlpha + ')';
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(mx - s,     my - s + t); ctx.lineTo(mx - s, my - s); ctx.lineTo(mx - s + t, my - s);
    ctx.moveTo(mx + s - t, my - s);     ctx.lineTo(mx + s, my - s); ctx.lineTo(mx + s,     my - s + t);
    ctx.moveTo(mx + s,     my + s - t); ctx.lineTo(mx + s, my + s); ctx.lineTo(mx + s - t, my + s);
    ctx.moveTo(mx - s + t, my + s);     ctx.lineTo(mx - s, my + s); ctx.lineTo(mx - s,     my + s - t);
    ctx.stroke();

    // Center dot
    ctx.beginPath();
    ctx.arc(mx, my, 1.5, 0, 6.2832);
    ctx.fillStyle = col + CFG.reticleAlpha + ')';
    ctx.fill();
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
