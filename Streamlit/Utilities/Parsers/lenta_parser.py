import asyncio
import csv
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
import dateparser
import pandas as pd
from bs4 import BeautifulSoup


def daterange(start_date: datetime, end_date: datetime):
    current = start_date.date()
    last = end_date.date()
    while current <= last:
        yield current
        current += timedelta(days=1)


async def fetch(session, url):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.text()
            return None
    except (
        TimeoutError,
        aiohttp.client_exceptions.ClientConnectorError,
        aiohttp.client_exceptions.ClientOSError,
        aiohttp.client_exceptions.ClientPayloadError,
    ):
        await asyncio.sleep(10)
        return await fetch(session, url)


def parse_page(page_content, source_date):
    soup = BeautifulSoup(page_content, "html.parser")
    rows = []
    for item in soup.find_all("li", class_="archive-page__item _news"):
        link_tag = item.find("a")
        title_tag = item.find("h3")
        time_tag = item.find("time")
        if link_tag is None or title_tag is None or time_tag is None:
            continue

        parsed_date = dateparser.parse(
            time_tag.get_text(strip=True),
            languages=["ru"],
            settings={"RELATIVE_BASE": datetime.combine(source_date, datetime.min.time())},
        )
        if parsed_date is None:
            continue

        url = link_tag.get("href", "")
        if url.startswith("/"):
            url = f"https://lenta.ru{url}"
        rows.append([parsed_date.strftime("%Y-%m-%d %H:%M:%S"), title_tag.get_text(strip=True), url])
    return rows


async def parse_day(session, source_date):
    rows = []
    page = 1
    while True:
        url = f"https://lenta.ru/{source_date.strftime('%Y/%m/%d')}/page/{page}/"
        page_content = await fetch(session, url)
        if not page_content:
            break

        page_rows = parse_page(page_content, source_date)
        if not page_rows:
            break

        rows.extend(page_rows)
        page += 1
    return rows


def normalize_csv(file_path):
    path = Path(file_path)
    if not path.exists() or path.stat().st_size == 0:
        pd.DataFrame(columns=["datetime", "title", "url"]).to_csv(path, index=False)
        return

    df = pd.read_csv(path)
    if df.empty:
        pd.DataFrame(columns=["datetime", "title", "url"]).to_csv(path, index=False)
        return

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).drop_duplicates()
    df = df.sort_values(by="datetime")
    df.to_csv(path, index=False)


async def main(start_date, end_date, file_path):
    rows = []
    async with aiohttp.ClientSession() as session:
        for current_date in daterange(start_date, end_date):
            rows.extend(await parse_day(session, current_date))

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with open(path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["datetime", "title", "url"])
        writer.writerows(rows)
    normalize_csv(path)


def parse(start_date, end_date, file_path):
    asyncio.run(main(start_date, end_date, file_path))
