// Records the controller demo against http://127.0.0.1:8787
// Needs Playwright plus system Chrome: node demo/record.mjs
import { createRequire } from "module";
import { mkdir } from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";

const require = createRequire("/tmp/pw-demo/package.json");
const { chromium } = require("playwright");

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)));
const rawDir = path.join(root, "video-raw");
await mkdir(rawDir, { recursive: true });

const browser = await chromium.launch({ channel: "chrome", headless: true });
const context = await browser.newContext({
  viewport: { width: 1440, height: 900 },
  deviceScaleFactor: 2,
  recordVideo: { dir: rawDir, size: { width: 1440, height: 900 } },
});
const page = await context.newPage();
await page.addInitScript(() => {
  const dot = document.createElement("div");
  dot.id = "demo-cursor";
  dot.style.cssText = [
    "position:fixed",
    "z-index:2147483647",
    "width:18px",
    "height:18px",
    "margin:-9px 0 0 -9px",
    "border-radius:50%",
    "background:#c24b1a",
    "box-shadow:0 0 0 6px rgba(194,75,26,0.25)",
    "pointer-events:none",
    "left:180px",
    "top:220px",
    "transition:left .45s ease, top .45s ease",
  ].join(";");
  document.addEventListener("DOMContentLoaded", () => document.body.appendChild(dot));
});

await page.goto("http://127.0.0.1:8787/", { waitUntil: "networkidle" });
await page.waitForTimeout(1600);

async function show(chip, heading) {
  const button = page.getByRole("button", { name: chip, exact: true });
  const box = await button.boundingBox();
  if (!box) throw new Error("missing chip " + chip);
  await page.evaluate(({ x, y }) => {
    const dot = document.getElementById("demo-cursor");
    if (!dot) return;
    dot.style.left = x + "px";
    dot.style.top = y + "px";
  }, { x: box.x + box.width / 2, y: box.y + box.height / 2 });
  await page.waitForTimeout(500);
  await button.click();
  await page.getByRole("heading", { level: 2, name: heading }).waitFor({ timeout: 20000 });
  await page.waitForTimeout(2400);
}

await show("Timer", "20 min");
await show("Message", "Meera");
await show("Weather", "Lisbon");
await show("Music", "Satellite");
await show("Directions", "Koregaon Park");
await show("Refuse", "No action");
await show("Outside list", "Rejected");
await page.waitForTimeout(900);

const video = page.video();
await context.close();
await browser.close();
const saved = await video.path();
console.log(saved);
