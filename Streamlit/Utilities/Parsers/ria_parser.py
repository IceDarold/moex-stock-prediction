import asyncio
import csv
from datetime import datetime, time, timedelta
from pathlib import Path

import aiohttp
import pandas as pd
from bs4 import BeautifulSoup


def daterange(start_date: datetime, end_date: datetime, reverse=False):
    current = start_date.date()
    last = end_date.date()
    dates = []
    while current <= last:
        dates.append(current)
        current += timedelta(days=1)
    return reversed(dates) if reverse else dates


async def fetch(session, url):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.text()
            return None
    except (TimeoutError, aiohttp.client_exceptions.ClientConnectorError, aiohttp.client_exceptions.ClientOSError):
        await asyncio.sleep(10)
        return await fetch(session, url)


async def parse_day(session, source_date):
    url = f"https://ria.ru/services/lenta/more.html?date={source_date.strftime('%Y%m%d')}T"
    rows = []
    formatted_time = time(23, 59, 59).strftime("%H%M%S")

    while True:
        page_content = await fetch(session, f"{url}{formatted_time}")
        if not page_content:
            break

        soup = BeautifulSoup(page_content, "html.parser")
        news_items = soup.find_all("div", class_="list-item")
        if not news_items:
            break

        reached_previous_day = False
        last_datetime = None
        for item in news_items:
            content = item.find("div", class_="list-item__content")
            info = item.find("div", class_="list-item__info")
            if content is None or info is None:
                continue

            title_tag = content.find("a", class_="list-item__title color-font-hover-only")
            date_tag = info.find("div", class_="list-item__date")
            if title_tag is None or date_tag is None:
                continue

            date_text = date_tag.get_text(strip=True).replace(" ", "")
            parts = date_text.split(",")
            if len(parts) == 2:
                try:
                    item_day = int(parts[0])
                except ValueError:
                    item_day = source_date.day
                if item_day != source_date.day:
                    reached_previous_day = True
                    break
                clock = parts[1]
            else:
                clock = parts[0]

            try:
                last_datetime = datetime.combine(source_date, datetime.strptime(clock, "%H:%M").time())
            except ValueError:
                continue

            rows.append([last_datetime.strftime("%Y-%m-%d %H:%M:%S"), title_tag.get_text(strip=True), title_tag["href"]])

        if reached_previous_day or last_datetime is None:
            break
        next_time = last_datetime.strftime("%H%M%S")
        if next_time == formatted_time:
            break
        formatted_time = next_time

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
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0

    async with aiohttp.ClientSession() as session:
        with open(path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["datetime", "title", "url"])
            for current_date in daterange(start_date, end_date, reverse=True):
                writer.writerows(await parse_day(session, current_date))
    normalize_csv(path)


def parse(start_date, end_date, file_path):
    asyncio.run(main(start_date, end_date, file_path))
