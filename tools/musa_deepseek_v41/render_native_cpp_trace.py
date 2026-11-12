# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render verified actual native C++/GPU intervals and entry COUNT in Perfetto UI."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger
from playwright.async_api import async_playwright


async def _render(args: argparse.Namespace, receipt: dict) -> None:
    scope = receipt.get("selected_scope") or receipt["selected_step"]
    begin = round(scope["ts"] * 1000)
    end = round((scope["ts"] + scope["dur"]) * 1000)
    async with async_playwright() as playwright:
        context = await playwright.chromium.launch_persistent_context(
            str(args.browser_profile),
            executable_path=args.chromium,
            headless=True,
            args=["--no-sandbox"],
            viewport={"width": 1900, "height": 1500},
        )
        page = await context.new_page()
        await page.goto(
            "https://ui.perfetto.dev", wait_until="networkidle", timeout=60000
        )
        await page.locator("input.trace_file").set_input_files(str(args.trace))
        await page.wait_for_function(
            "window.app && window.app._activeTrace && "
            "window.app._activeTrace.workspaces.currentWorkspace.flatTracksOrdered"
            ".some(n => n.uri && /stack_samples/.test(n.uri)) && "
            "window.app._activeTrace.workspaces.currentWorkspace.flatTracksOrdered"
            ".filter(n => n.uri && /track_event_/.test(n.uri)).length >= 2",
            timeout=120000,
        )
        await page.evaluate(
            """window.app.commands.registerCommand({
          id:'xiaopang.expand_native',name:'Expand verified actual native trace',
          callback:()=>{for(let node of window.app._activeTrace.workspaces.currentWorkspace.children)node.expandAll()}})"""
        )
        await page.evaluate("window.app.commands.runCommand('xiaopang.expand_native')")
        await page.keyboard.press("Escape")
        cookie = page.get_by_role("button", name="OK", exact=True)
        if await cookie.count():
            await cookie.click()
        await page.wait_for_timeout(1500)
        await page.screenshot(path=str(args.output / "真实CPU区间与GPU时间线.png"))
        script = """window.app.commands.registerCommand({
          id:'xiaopang.select_native',name:'Select actual native entry counts',
          callback:()=>{const trace=window.app._activeTrace;
            const uris=trace.workspaces.currentWorkspace.flatTracksOrdered
              .filter(n=>n.uri&&/stack_samples/.test(n.uri)).map(n=>n.uri);
            trace.selection.selectArea({start:BEGIN_NS,end:END_NS,trackUris:uris});}})"""
        script = script.replace("BEGIN_NS", str(begin) + "n").replace(
            "END_NS", str(end) + "n"
        )
        await page.evaluate(script)
        await page.evaluate("window.app.commands.runCommand('xiaopang.select_native')")
        await page.wait_for_timeout(2000)
        await page.screenshot(
            path=str(args.output / "实际C++回溯_ENTRY_COUNT火焰图.png")
        )
        (args.output / "Perfetto页面文本.txt").write_text(
            await page.locator("body").inner_text()
        )
        await page.get_by_text("vertical_align_top", exact=True).click()
        await page.wait_for_timeout(1500)
        await page.screenshot(
            path=str(args.output / "完整实际C++调用链_ENTRY_COUNT.png")
        )
        tracks = await page.evaluate(
            "window.app._activeTrace.workspaces.currentWorkspace.flatTracksOrdered.map(n=>({name:n.name,uri:n.uri}))"
        )
        (args.output / "Perfetto实际轨道.json").write_text(json.dumps(tracks, indent=2))
        await context.close()
    (args.output / "rendered.json").write_text(
        json.dumps(
            {
                "status": "RENDERED_REAL_INTERVALS_AND_NATIVE_ENTRY_COUNT_WITH_PERFETTO_UI",
                "trace": str(args.trace),
                "output_sha256": receipt["output_sha256"],
                "native_entry_samples": receipt["native_entry_samples"],
                "selected_cpu_interval": scope,
                "boundary": "Entry counts identify real C++ callers; widths are not CPU-time samples. CPU/GPU interval timings are copied from the actual Kineto trace.",
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--browser-profile", type=Path, required=True)
    parser.add_argument("--chromium", default="/usr/bin/chromium")
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    if (
        receipt["status"] != "CONVERTED_AND_VERIFIED"
        or receipt["verification"]["errors"]
    ):
        raise ValueError("Only render a Trace Processor verified artifact")
    if hashlib.sha256(args.trace.read_bytes()).hexdigest() != receipt["output_sha256"]:
        raise ValueError("Trace changed after verification")
    args.output.mkdir(parents=True, exist_ok=False)
    asyncio.run(_render(args, receipt))
    logger.info("Rendered actual native trace and COUNT profile: %s", args.output)


if __name__ == "__main__":
    main()
