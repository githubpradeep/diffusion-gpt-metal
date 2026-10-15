// Records the photograph page against http://127.0.0.1:8787/draw
// Needs Playwright plus system Chrome: node demo/record-draw.mjs
import { createRequire } from "module";
import { mkdir } from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";

const require = createRequire("/tmp/pw-demo/package.json");
const { chromium } = require("playwright");

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)));
const rawDir = path.join(root, "video-raw");
await mkdir(rawDir, { recursive: true });

const width = 1440;
const height = 1380;
const browser = await chromium.launch({ channel: "chrome", headless: true });
const context = await browser.newContext({
  viewport: { width, height },
  recordVideo: { dir: rawDir, size: { width, height } },
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

await page.goto("http://127.0.0.1:8787/draw", { waitUntil: "networkidle" });
await page.getByRole("button", { name: "house", exact: true }).waitFor();
await page.waitForTimeout(1200);

async function show(name) {
  const button = page.getByRole("button", { name, exact: true });
  const box = await button.boundingBox();
  if (!box) throw new Error("missing chip " + name);
  await page.evaluate(({ x, y }) => {
    const dot = document.getElementById("demo-cursor");
    if (!dot) return;
    dot.style.left = x + "px";
    dot.style.top = y + "px";
  }, { x: box.x + box.width / 2, y: box.y + box.height / 2 });
  await page.waitForTimeout(450);
  await button.click();
  await page.locator("#ar-status").getByText(name + " · 65 frames").waitFor({ timeout: 120000 });
  await page.locator("#dit-status").getByText(name + " · 20 frames").waitFor({ timeout: 120000 });
  await page.locator("#ncm-status").getByText(name + " · 6 frames").waitFor({ timeout: 120000 });
  // 65 patch frames at 70ms, plus a beat on the finished picture.
  // The coarse-to-fine run is six frames at 280ms, so it finishes inside this hold.
  await page.waitForTimeout(5200);
}

await show("house");
await show("cat");
await show("pizza");
await page.waitForTimeout(800);

const video = page.video();
await context.close();
await browser.close();
const saved = await video.path();
console.log(saved);
