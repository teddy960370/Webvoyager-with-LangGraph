import json
import os
import argparse
from firecrawl import FirecrawlApp, ScrapeOptions ,AsyncFirecrawlApp
from langchain_openai import AzureOpenAI,OpenAI,AzureChatOpenAI,ChatOpenAI


transform_manual_prompt = """
你是一個專業分析網頁內容並生成網頁操作指南的機器人。根據提供的網頁內容Markdown和部分截圖，分析每個網站的使用方法，並撰寫以Markdown為基礎的網站使用指南。

輸入格式：列表，其中包含多個字典，每個字典代表一個網頁。

- `isHome`: Boolean，表示該頁是否是首頁。
- `url`: 字符串，網頁的URL。
- `title`: 字符串，網頁的標題 (可以為空)。
- `description`: 字符串，網頁的描述 (可以為空)。
- `markdown`: 字符串，網頁的Markdown內容 (用於分析使用)。

# 步驟

1. **分析首頁**: 確認首頁URL，分析其`markdown`內容以生成網站總體概述。
2. **分析各非首頁頁面**: 每個非首頁頁面的`url`應該有自己獨立的分析。
    - 描述頁面的內容和功能。
    - 指導用戶如何從首頁導航到這些頁面。
3. **整合信息**: 將分析過的內容整合為一份全面的使用指南文檔。

# 輸出格式

此使用指南應以Markdown格式呈現，包含以下結構：

# 文件標題: 此份文件的內容標題

## 內容概述
- 概述: 此網站的內容概述

## 各頁面功能
1. **首頁**
   - URL: `[首頁URL]`
   - 主要功能: [對首頁的功能進行簡要描述]

2. **其他頁面**
   - URL: `[頁面URL]`
   - 內容與功能: [對頁面內容和功能的詳細介紹]
   - 導航: 從首頁到達此頁的步驟

# 範例

**Input**

```json
[
	{
		"isHome" : true,
		"url" : "https://exampleHome.com/",
		"title" : "Home",
		"description" : "This is the homepage.",
		"markdown" : "Welcome to the homepage, your starting point to explore."
	},
	{
		"isHome" : false,
		"url" : "https://example.com/contact",
		"title" : "Contact",
		"description" : "Contact us page.",
		"markdown" : "Find our contact details here."
	}
]
```

**Output**

# 文件標題: 網站使用指南

## 內容概述
- 概述: 這個網站是一個展示其產品和服務的起點頁面，並提供了聯繫資訊。

## 各頁面功能
1. **首頁**
   - URL: `https://exampleHome.com/`
   - 主要功能: 主頁歡迎用戶並提供網站的總覽。

2. **聯繫頁面**
   - URL: `https://example.com/contact`
   - 內容與功能: 提供聯繫方式和表單供用戶交互。
   - 導航: 用戶可以從首頁通過網站頂部的導航欄訪問聯繫頁。


# 注意事項

- 請确保每個頁面的分析內容清晰且完整。
- 確保導航指引具體且可操作。
- 輸出中的所有Markdown需符合標準，確保格式正確。

"""

def get_manual(llm, markdown_content_list):

    # Convert the markdown_content_list to the required format
    formatted_content = []
    
    for item in markdown_content_list:
        formatted_item = {
            "isHome": item['doc_id'] == 0, # Assuming the first item is the homepage
            "url": item['url'],
            "title": item.get('title', ''),
            "description": item.get('description', ''),
            "markdown": item.get('markdown', '')
        }
        formatted_content.append(formatted_item)
    
    user_prompt = json.dumps(formatted_content, ensure_ascii=False, indent=2)
    
    messages = [
        {"role": "system", "content": transform_manual_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = llm.invoke(messages)

    return response.content

def get_md_from_web(app, url_list):
    scrape_options = ScrapeOptions(
        #includeTags=None,  # 可選：用於包含特定路徑的正規表示式模式
        #excludeTags=None,  # 可選：用於排除特定路徑的正規表示式模式
        #blockAds=True,  # 可選：阻止廣告
        #onlyMainContent=True,  # 可選：僅爬取主要內容
        formats=['markdown', 'screenshot'],  # 可選：指定返回的格式為 URL 列表
        #removeBase64Images=True,  # 可選：移除 Base64 圖像
    )

    doc = []
    doc_count = 0
    for url in url_list:
        # 執行爬取
        #crawled_data = app.crawl_url(
        #    url=url,
        #    limit=1,
        #    max_depth=0,
        #    scrape_options=scrape_options
        #)

        scraped_data = app.scrape_url(
            url=url,
            formats=['markdown', 'screenshot'],  # 可選：指定返回的格式為 URL 列表
            removeBase64Images=True,
            blockAds=True,  # 可選：阻止廣告
            #scrape_options=scrape_options
        )

        if scraped_data:
            markdown = scraped_data.markdown
            screenshot_url = scraped_data.screenshot
            metadata = scraped_data.metadata
            # 如果有截圖，則下載並轉換為 Base64
            if screenshot_url:
                import requests
                from base64 import b64encode

                response = requests.get(screenshot_url)
                if response.status_code == 200:
                    screenshot = b64encode(response.content).decode('utf-8')
                else:
                    screenshot = ''
                    
            doc.append({
                'doc_id': doc_count,
                'url': url,
                'title': metadata.get('title', 'No Title'),
                'description': metadata.get('description', 'No Description'),
                'language': metadata.get('language', ''),
                'markdown': markdown,
                'screenshot_url': screenshot_url,
                'llm_desc': ''
            })
            doc_count += 1
        else:
            print("未發現 URL 或回應格式非預期。")

        # 爬取網站後間隔60秒，以避免過於頻繁的請求
        import time
        time.sleep(30)
    
    return doc

def save_to_json(webname, data):

    output_dir = "./AutoManual/results/" + webname
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, webname + '_manual.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='./AutoManual/target.json')
    parser.add_argument("--api_key", default="key", type=str, help="YOUR_OPENAI_API_KEY")
    parser.add_argument("--api_model", default="gpt-4-vision-preview", type=str, help="api model name")
    parser.add_argument("--temperature", type=float, default=1.0)

    args = parser.parse_args()

    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=args.api_key,
        model=args.api_model,
        temperature=args.temperature
    )

    # 使用您的 Firecrawl API 金鑰進行初始化
    app = FirecrawlApp(api_key="fc-567a5e213b604637bee964366187e08d")

    webs = ['Amazon', 'Apple', 'ArXiv', 'BBC News', 'Cambridge Dictionary',
            'Coursera', 'ESPN', 'GitHub', 'Google Map', 'Huggingface', 'Wolfram Alpha','Booking','Google Flights', 'Google Search']
    webs = ['Google Flights']
    # read url list from file
    web_list = []
    with open(args.test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            web_list.append(
                (item['web_name'], item['urls'])
            )

    for web in webs :
        if web not in [item[0] for item in web_list]:
            print(f"Web {web} not found in the test file.")
            continue

        url_list = [url for name, urls in web_list if name == web for url in urls]

        doc = get_md_from_web(app, url_list)

        output_dir = "./AutoManual/results/" + web
        # 生成使用手冊
        automanual = ''
        automanual = get_manual(llm , doc)
        manual_output_file = os.path.join(output_dir, web + '_manual.md')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(manual_output_file, 'w', encoding='utf-8') as f:
            f.write(automanual)

if __name__ == "__main__":
    main()



