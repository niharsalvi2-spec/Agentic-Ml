# Web Scraping

Automatically extracting data from websites when no API exists.

## Choosing a tool

| Situation | Tool |
|---|---|
| Static HTML, small/medium site | BeautifulSoup |
| Thousands of pages, need speed | Scrapy |
| Content rendered by JavaScript | Selenium |

| | BeautifulSoup | Scrapy |
|---|---|---|
| Speed | Slow | Fast (async) |
| Scale | Small sites | Thousands of pages |
| Learning curve | Easy | Moderate |
| Built-in storage | No | Yes (pipelines) |

## Always check first

- `https://<site>/robots.txt` — `Disallow: /` means do not scrape that path.
- The site's Terms of Service for scraping restrictions.
- Whether the target data includes personal data (GDPR/DPDP apply even to scraped data).
- Add delays between requests (`time.sleep(1-2)`); never hammer a server.

## BeautifulSoup — static sites

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

url = "https://books.toscrape.com/"
response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
soup = BeautifulSoup(response.text, "html.parser")

books = soup.find_all("article", class_="product_pod")

data = []
for book in books:
    data.append({
        "title": book.h3.a["title"],
        "price": book.find("p", class_="price_color").text,
        "rating": book.p["class"][1],
    })
    time.sleep(0.5)  # be polite

df = pd.DataFrame(data)
df.to_csv("books.csv", index=False)
```

Key methods: `find('tag')`, `find_all('tag')`, `find('tag', class_='x')`, `tag['attribute']`, `tag.text`.

## Scrapy — large scale

```python
# scrapy startproject myproject
# myproject/spiders/books_spider.py
import scrapy

class BooksSpider(scrapy.Spider):
    name = "books"
    start_urls = ["https://books.toscrape.com/"]
    custom_settings = {"DOWNLOAD_DELAY": 1, "CONCURRENT_REQUESTS": 4}

    def parse(self, response):
        for book in response.css("article.product_pod"):
            yield {
                "title": book.css("h3 a::attr(title)").get(),
                "price": book.css(".price_color::text").get(),
                "rating": book.css("p.star-rating::attr(class)").get(),
            }
        next_page = response.css("li.next a::attr(href)").get()
        if next_page:
            yield response.follow(next_page, self.parse)

# scrapy crawl books -o books.json
```

## Selenium — JavaScript-rendered sites

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd

options = webdriver.ChromeOptions()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)

driver.get("https://quotes.toscrape.com/js/")
wait = WebDriverWait(driver, 10)
wait.until(EC.presence_of_element_located((By.CLASS_NAME, "quote")))

quotes = driver.find_elements(By.CLASS_NAME, "quote")
data = [{
    "quote": q.find_element(By.CLASS_NAME, "text").text,
    "author": q.find_element(By.CLASS_NAME, "author").text,
} for q in quotes]

driver.quit()
df = pd.DataFrame(data)
```

Common Selenium actions: `element.click()`, `element.send_keys('text')`,
`driver.execute_script("window.scrollBy(0,500)")`, `driver.save_screenshot('page.png')`,
`element.get_attribute('href')`.

## Image / audio scraping

```python
# pip install icrawler
from icrawler.builtin import GoogleImageCrawler
crawler = GoogleImageCrawler(storage={"root_dir": "./images/cats"})
crawler.crawl(keyword="cat", max_num=500)
```

```python
# pip install yt-dlp — audio for speech datasets
import yt_dlp
ydl_opts = {"format": "bestaudio/best", "outtmpl": "./audio/%(title)s.%(ext)s"}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download(["https://youtube.com/watch?v=xxxxx"])
```

See `code-generation.md` for the retry/backoff wrapper — apply it to every `requests.get`/Selenium call above rather than calling them bare.
