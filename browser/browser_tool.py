import asyncio
from playwright.async_api import async_playwright


async def _browse(url: str, max_chars: int = 4000) -> str:

    async with async_playwright() as p:

        browser = await p.chromium.launch(
            headless=True
        )

        page = await browser.new_page(
            viewport={
                "width": 1280,
                "height": 720
            }
        )

        try:

            await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=30000
            )

            try:
                await page.wait_for_load_state(
                    "networkidle",
                    timeout=10000
                )
            except Exception:
                pass

            content = await page.locator(
                "body"
            ).inner_text()

            content = " ".join(
                content.split()
            )

            return content[:max_chars]

        except Exception as e:

            return f"Browser error while opening {url}: {str(e)}"

        finally:

            await browser.close()


def browse_webpage(url: str) -> str:
    """
    Synchronous wrapper.

    This makes the browser easy to use
    from the existing LangChain agent.
    """

    try:
        return asyncio.run(
            _browse(url)
        )

    except RuntimeError:

        # Handles environments with an
        # existing event loop.
        loop = asyncio.new_event_loop()

        try:
            return loop.run_until_complete(
                _browse(url)
            )
        finally:
            loop.close()