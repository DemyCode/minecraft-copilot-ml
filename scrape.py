"""
Scraper for minecraft-schematics.com.

Single-threaded, one Firefox instance, simple loop.

Usage:
    nix-shell -p geckodriver --run \
        "uv run python scrape.py --out_dir /home/nixos/schematics_scraped"

    # Resume from a specific ID:
    nix-shell -p geckodriver --run \
        "uv run python scrape.py --out_dir /home/nixos/schematics_scraped --start 5000"
"""

import argparse
import logging
import os
import random
import re
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.common.exceptions import TimeoutException
from tqdm import tqdm

BASE_URL = "https://www.minecraft-schematics.com"
FIREFOX_BIN = "/run/current-system/sw/bin/firefox"
FIREFOX_PROFILE = Path("/home/nixos/.mozilla/firefox/2e0gkbbm.default")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def read_firefox_cookies() -> list[dict]:
    db = FIREFOX_PROFILE / "cookies.sqlite"
    tmp = tempfile.mktemp(suffix=".sqlite")
    shutil.copy2(db, tmp)
    try:
        conn = sqlite3.connect(tmp)
        rows = conn.execute(
            "SELECT name, value, host FROM moz_cookies WHERE host LIKE '%minecraft-schematics%'"
        ).fetchall()
        conn.close()
    finally:
        os.unlink(tmp)
    return [{"name": r[0], "value": r[1], "domain": r[2]} for r in rows]


def make_driver(dl_dir: Path) -> webdriver.Firefox:
    svc = Service(executable_path=shutil.which("geckodriver"), log_output="/dev/null")
    opts = Options()
    opts.add_argument("-headless")
    opts.binary_location = FIREFOX_BIN
    opts.set_preference("browser.download.folderList", 2)
    opts.set_preference("browser.download.dir", str(dl_dir))
    opts.set_preference("browser.download.useDownloadDir", True)
    opts.set_preference("browser.download.manager.showWhenStarting", False)
    opts.set_preference("browser.helperApps.neverAsk.saveToDisk",
                        "application/octet-stream,application/zip,application/x-zip-compressed,application/x-zip")
    opts.set_preference("pdfjs.disabled", True)
    opts.set_preference("dom.webdriver.enabled", False)
    opts.set_preference("useAutomationExtension", False)

    driver = webdriver.Firefox(service=svc, options=opts)
    driver.set_page_load_timeout(30)

    driver.get(BASE_URL)
    for c in read_firefox_cookies():
        try:
            driver.add_cookie(c)
        except Exception:
            pass
    driver.get(BASE_URL)

    if "logout" not in driver.page_source.lower():
        driver.quit()
        raise RuntimeError("Could not authenticate — make sure you are logged in to minecraft-schematics.com in Firefox.")

    log.info("Authenticated.")
    return driver


def _wait_for_cf(driver: webdriver.Firefox, timeout: int = 10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if "just a moment" not in driver.title.lower():
            return
        time.sleep(0.5)


def _wait_for_download(dl_dir: Path, before: set, timeout: int = 25) -> Path | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.5)
        new = set(dl_dir.glob("*")) - before
        complete = [f for f in new if f.suffix not in (".part", ".crdownload", ".tmp")]
        if complete:
            return max(complete, key=lambda f: f.stat().st_mtime)
    return None


_KNOWN_EXTS = (".schematic", ".schem", ".zip", ".nbt")

def download_one(driver: webdriver.Firefox, schematic_id: int, out_dir: Path, dl_dir: Path, delay: float) -> str:
    if any((out_dir / f"{schematic_id}{ext}").exists() for ext in _KNOWN_EXTS):
        return "skip"

    time.sleep(delay + random.uniform(0, delay * 0.5))

    try:
        driver.get(f"{BASE_URL}/schematic/{schematic_id}/")
        _wait_for_cf(driver)
        if "login" in driver.current_url:
            return "error:auth_lost"
        if "404" in driver.title or "not found" in driver.title.lower():
            return "missing"

        dl_file = None
        for fmt in ("schematic", "world_save"):
            before = set(dl_dir.glob("*"))
            try:
                driver.get(f"{BASE_URL}/schematic/{schematic_id}/download/action/?type={fmt}")
            except TimeoutException:
                pass
            _wait_for_cf(driver)
            if "login" in driver.current_url:
                return "error:auth_lost"
            dl_file = _wait_for_download(dl_dir, before, timeout=15)
            if dl_file is not None:
                break

        if dl_file is None:
            return "missing"

        ext = dl_file.suffix or ".schematic"
        target = out_dir / f"{schematic_id}{ext}"
        shutil.move(str(dl_file), target)
        return "ok"

    except Exception as e:
        return f"error:{e}"


def scrape(out_dir: Path, start: int, end: int, delay: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    dl_dir = out_dir / ".dl_tmp"
    dl_dir.mkdir(exist_ok=True)

    driver = make_driver(dl_dir)
    stats = {"ok": 0, "skip": 0, "missing": 0, "error": 0}

    try:
        for schematic_id in tqdm(range(start, end + 1), unit="schematic"):
            result = download_one(driver, schematic_id, out_dir, dl_dir, delay)

            key = result if result in ("ok", "skip", "missing") else "error"
            if key == "error":
                log.warning("ID %d: %s", schematic_id, result)
            stats[key] += 1

            have = stats["ok"] + stats["skip"]
            scanned = have + stats["missing"]
            projected = int(have / scanned * (scanned + (end - schematic_id))) if scanned > 0 else "?"
            tqdm.write("") if False else None  # keep import
            tqdm.get_lock()
            print(f"\r{stats}  projected={projected}", end="", flush=True)

    finally:
        driver.quit()
        try:
            dl_dir.rmdir()
        except Exception:
            pass

    log.info("Done. %s", stats)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="schematics_scraped")
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=30000)
    p.add_argument("--delay", type=float, default=2.0)
    args = p.parse_args()

    scrape(
        out_dir=Path(args.out_dir),
        start=args.start,
        end=args.end,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
