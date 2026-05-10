import asyncio
import csv
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
import pandas as pd
import requests
from bs4 import BeautifulSoup

REQUEST_TIMEOUT_SECONDS = 20
MAX_FETCH_RETRIES = 2
MAX_ARCHIVE_PAGES = 20
USER_AGENT = "Mozilla/5.0 (compatible; moex-stock-prediction/1.0)"

try:
    import dateparser
except ModuleNotFoundError:
    dateparser = None


MONTHS = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}


def daterange(start_date: datetime, end_date: datetime):
    current = start_date.date()
    last = end_date.date()
    while current <= last:
        yield current
        current += timedelta(days=1)


def fetch_with_requests(url):
    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers={"User-Agent": USER_AGENT},
        )
    except requests.exceptions.RequestException:
        return None

    if response.status_code != 200:
        return None
    return response.text


async def fetch(session, url):
    for attempt in range(MAX_FETCH_RETRIES + 1):
        page_content = await asyncio.to_thread(fetch_with_requests, url)
        if page_content:
            return page_content
        if attempt < MAX_FETCH_RETRIES:
            await asyncio.sleep(2)
    return None


def parse_lenta_datetime(raw_text, source_date):
    if dateparser is not None:
        parsed_date = dateparser.parse(
            raw_text,
            languages=["ru"],
            settings={"RELATIVE_BASE": datetime.combine(source_date, datetime.min.time())},
        )
        if parsed_date is not None:
            return parsed_date

    normalized = raw_text.replace(",", " ").split()
    if len(normalized) < 4:
        return None

    try:
        hour, minute = map(int, normalized[0].split(":"))
        day = int(normalized[1])
        month = MONTHS[normalized[2].lower()]
        year = int(normalized[3])
    except (KeyError, TypeError, ValueError):
        return None

    return datetime(year, month, day, hour, minute)


def parse_page(page_content, source_date):
    soup = BeautifulSoup(page_content, "html.parser")
    rows = []
    for item in soup.find_all("li", class_="archive-page__item _news"):
        link_tag = item.find("a")
        title_tag = item.find("h3")
        time_tag = item.find("time")
        if link_tag is None or title_tag is None or time_tag is None:
            continue

        parsed_date = parse_lenta_datetime(time_tag.get_text(strip=True), source_date)
        if parsed_date is None:
            continue

        url = link_tag.get("href", "")
        if url.startswith("/"):
            url = f"https://lenta.ru{url}"
        rows.append([parsed_date.strftime("%Y-%m-%d %H:%M:%S"), title_tag.get_text(strip=True), url])
    return rows


async def parse_day(session, source_date):
    rows = []
    for page in range(1, MAX_ARCHIVE_PAGES + 1):
        url = f"https://lenta.ru/{source_date.strftime('%Y/%m/%d')}/page/{page}/"
        page_content = await fetch(session, url)
        if not page_content:
            break

        page_rows = parse_page(page_content, source_date)
        if not page_rows:
            break

        rows.extend(page_rows)
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
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": USER_AGENT}) as session:
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
