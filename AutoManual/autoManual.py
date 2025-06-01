import argparse
from firecrawl import FirecrawlApp, ScrapeOptions
from langchain_openai import AzureOpenAI,OpenAI,AzureChatOpenAI,ChatOpenAI



def get_manual(llm , markdown_content_list):
    print("依據markdown撰寫使用手冊")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='data/test.json')
    parser.add_argument('--max_iter', type=int, default=5)
    parser.add_argument("--api_key", default="key", type=str, help="YOUR_OPENAI_API_KEY")
    parser.add_argument("--api_model", default="gpt-4-vision-preview", type=str, help="api model name")
    parser.add_argument("--output_dir", type=str, default='results')
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_attached_imgs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--download_dir", type=str, default="downloads")
    parser.add_argument("--text_only", action='store_true')
    # for web browser
    parser.add_argument("--headless", action='store_true', help='The window of selenium')
    parser.add_argument("--save_accessibility_tree", action='store_true')
    parser.add_argument("--force_device_scale", action='store_true')
    parser.add_argument("--window_width", type=int, default=1024)
    parser.add_argument("--window_height", type=int, default=768)  # for headless mode, there is no address bar
    parser.add_argument("--fix_box_color", action='store_true')
    parser.add_argument("--azure_endpoint", type=str, default="")
    parser.add_argument("--api_version", type=str, default="")
    parser.add_argument("--use_rag", type=bool, default=False, help="Use RAG to get context for the task")
    parser.add_argument("--llm", type=str, default="openai", choices=["openai", "azure","openrouter","gemini"])
    parser.add_argument("--som_scan_all", type=bool, default=False)

    args = parser.parse_args()

    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=args.api_key,
        model=args.api_model,
        temperature=args.temperature
    )

    # 使用您的 Firecrawl API 金鑰進行初始化
    app = FirecrawlApp(api_key="fc-567a5e213b604637bee964366187e08d")

    scrape_options = ScrapeOptions(
        returnOnlyUrls=True,  # 關鍵：僅獲取 URL
        #includes=None,  # 可選：用於包含特定路徑的正規表示式模式
        #excludes=None,  # 可選：用於排除特定路徑的正規表示式模式
        #limit=100,  # 可選：返回的最大頁面數
        blockAds=True,  # 可選：阻止廣告
        onlyMainContent=True,  # 可選：僅爬取主要內容
        formats=['markdown','screenshot'],  # 可選：指定返回的格式為 URL 列表
        removeBase64Images=True,  # 可選：移除 Base64 圖像
    )

    url_list = [
        'https://global.espn.com/',
        'https://global.espn.com/nba/',
        'https://global.espn.com/nba/teams',
        'https://global.espn.com/nba/scoreboard',
        'https://global.espn.com/nba/schedule',
        'https://global.espn.com/nfl/',
        'https://global.espn.com/nfl/draft/rounds',
    ]

    doc = []
    doc_count = 0
    for url in url_list:
        # 執行爬取
        crawled_data = app.crawl_url(
            url=url,
            limit=1,
            max_depth=0,
            scrape_options=scrape_options
        )

        if crawled_data and crawled_data.completed:
            markdown = crawled_data.data[0].get('markdown', '')
            screenshot_url = crawled_data.data[0].get('screenshot', '')
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
                'markdown': markdown,
                'screenshot': screenshot,
                'llm_desc': ''
            })
            doc_count += 1
        else:
            print("未發現 URL 或回應格式非預期。")

    summary_markdown = ''
    prompt = "你是一個專業的網頁使用手冊撰寫者，請根據以下內容撰寫一份使用手冊。"
    for item in doc:
        # 由於markdown太大，分批使用LLM進行資料擷取
        response = llm.invoke(prompt + item['markdown'] + item['screenshot'])
        item['llm_desc'] = response.content
        summary_markdown += """Document ID: {doc_id}\nURL: {url}\n\n{llm_desc}\n\n""".format(
            doc_id=item['doc_id'],
            url=item['url'],
            llm_desc=item['llm_desc']
        )

    # 請求LLM整合所有 markdown 內容
    final_response = llm.invoke("請將以下內容整合成一份完整的使用手冊：\n\n" + summary_markdown)
    print("最終生成的使用手冊內容：")
    print(final_response.content)

if __name__ == "__main__":
    main()



