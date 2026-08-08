/* Loading the exports, and saying out loud how old they are.
 *
 * A static page has no idea whether the JSON behind it was written a minute ago
 * or the last time the scheduler worked. That is the failure mode worth
 * designing against here: not a wrong number, but a stale number wearing the
 * clothes of a live one. So every page loads through here, refreshes itself
 * while it is open, and states the age of what it is showing.
 */

/* Two layouts resolve through the same relative path:
 *   deployed bundle   /index.html            -> data/exports/
 *   repo served whole /site/index.html       -> ../data/exports/
 * The local runner mounts the exports at /data/exports/ so the first candidate
 * wins there too; the second exists for `python -m http.server` at the root. */
const CANDIDATES = ["data/exports/", "../data/exports/"];
let base = null;

export async function loadExport(name) {
  const bases = base ? [base] : CANDIDATES;
  let lastError;
  for (const b of bases) {
    try {
      const res = await fetch(`${b}${name}?t=${Date.now()}`, { cache: "no-store" });
      if (!res.ok) throw new Error(`${name}: ${res.status}`);
      const json = await res.json();
      base = b;
      return json;
    } catch (err) {
      lastError = err;
    }
  }
  throw lastError || new Error(`${name}: unreachable`);
}

/* ---------- freshness ---------- */

const MINUTE = 60000;

export function ago(iso) {
  if (!iso) return null;
  const then = new Date(iso).getTime();
  if (!Number.isFinite(then)) return null;
  return Math.max(0, Date.now() - then);
}

export function humanAge(ms) {
  if (ms === null || ms === undefined) return "unknown";
  const mins = Math.round(ms / MINUTE);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min ago`;
  const hours = Math.round(mins / 60);
  if (hours < 48) return `${hours} h ago`;
  return `${Math.round(hours / 24)} d ago`;
}

/* `expectedEveryMin` is the update cadence the deployment promises. Twice that
 * is late; eight times it is broken. The thresholds are deliberately generous:
 * a scheduled job that is merely queued should not be reported as an outage. */
export function freshness(payload, expectedEveryMin = 30) {
  const cov = payload.coverage || {};
  const builtMs = ago(payload.generated_at);
  const dataMs = ago(cov.latest_post_utc);
  const polledMs = ago(cov.last_poll_utc || cov.last_poll_attempt_utc);

  const late = expectedEveryMin * 2 * MINUTE;
  const broken = expectedEveryMin * 8 * MINUTE;

  let level = "ok";
  let why = "";
  if (builtMs === null) {
    level = "warn";
    why = "the build time is missing";
  } else if (builtMs > broken || (polledMs !== null && polledMs > broken)) {
    level = "bad";
    why = "the updater has not run for a while — treat these numbers as history";
  } else if (builtMs > late || (polledMs !== null && polledMs > late)) {
    level = "warn";
    why = "the last update is overdue";
  } else if (cov.last_poll_complete === false) {
    level = "warn";
    why = "the last poll did not finish closing its gap; the newest days may be short";
  }

  const parts = [`Rebuilt ${humanAge(builtMs)}`];
  if (polledMs !== null) parts.push(`source checked ${humanAge(polledMs)}`);
  if (dataMs !== null) parts.push(`newest post ${humanAge(dataMs)}`);

  return { level, text: parts.join(" · "), why, builtMs, dataMs, polledMs };
}

export function renderFreshness(el, payload, expectedEveryMin = 30) {
  if (!el) return;
  const f = freshness(payload, expectedEveryMin);
  el.className = `freshness ${f.level}`;
  el.innerHTML = f.why
    ? `<b>${f.text}.</b> ${f.why}.`
    : `${f.text}.`;
}

/* ---------- refresh loop ---------- */

/* Re-fetch while the tab is open, but only while it is actually being looked
 * at: a page left in a background tab for a week should not keep asking. */
export function autoRefresh(fn, seconds = 60) {
  let timer = null;
  const tick = async () => {
    if (document.visibilityState === "visible") {
      try {
        await fn();
      } catch {
        /* a failed refresh keeps the last good render rather than blanking it */
      }
    }
    timer = setTimeout(tick, seconds * 1000);
  };
  timer = setTimeout(tick, seconds * 1000);

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      clearTimeout(timer);
      tick();
    }
  });
  return () => clearTimeout(timer);
}
