import json
import re
import datetime
import random
import os
import pandas as pd
import argparse
from dateutil import parser
from dateutil.relativedelta import relativedelta

class TemplateReplacer:
    def __init__(self, json_path):
        self.json_path = json_path
        self.data = self._read_json()

    def _read_json(self):
        """讀取JSON檔案"""
        with open(self.json_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        data = []
        for line in lines:
            if line.strip():
                data.append(json.loads(line))
        return data

    def replaced_dates(self, seed=None, min_days=30, max_days=180):
        """替換問題中的日期模板為未來半年內的隨機日期，以及替換Apple產品名稱"""
        if seed is not None:
            random.seed(seed)
        
        # 取得當前日期
        today = datetime.datetime.now()
        
        # 複製資料以避免修改原始資料
        replaced_data = []
        
        # 定義正則表達式模式來匹配模板和可能的日期偏移
        template_pattern = r'(YYYY_MM_DD_TEMPLATE_PAST|YYYY_YY_TEMPLATE_PAST|YYYY_TEMPLATE|MM_TEMPLATE|DD_TEMPLATE|YYYY_MM_TEMPLATE|MM_DD_TEMPLATE|YYYY_MM_DD_TEMPLATE)([+-]\d+)?'

        for item in self.data:
            question = item.get('ques', '')
            
            # 檢查是否有任何模板匹配
            if re.search(template_pattern, question):
                # 生成一個未來隨機天數作為基準日期
                random_count = random.randint(min_days, max_days)
                base_date = today + datetime.timedelta(days=random_count)

                # 找到所有模板匹配並替換
                def replace_template(match):
                    template = match.group(1)
                    offset_str = match.group(2) or ""
                    
                    # 解析日期偏移量
                    offset_days = 0
                    if offset_str:
                        offset_days = int(offset_str)
                    
                    # 計算最終日期
                    final_date = base_date + datetime.timedelta(days=offset_days)
                    
                    # 根據模板類型，格式化日期
                    if template == 'YYYY_TEMPLATE':
                        return final_date.strftime('%Y')
                    elif template == 'MM_TEMPLATE':
                        return final_date.strftime('%m')
                    elif template == 'DD_TEMPLATE':
                        return final_date.strftime('%d')
                    elif template == 'YYYY_MM_TEMPLATE':
                        return final_date.strftime('%B, %Y')
                    elif template == 'MM_DD_TEMPLATE':
                        return final_date.strftime('%B %d')
                    elif template == 'YYYY_MM_DD_TEMPLATE':
                        return final_date.strftime('%B %d, %Y')
                    elif template == 'YYYY_MM_DD_TEMPLATE_PAST':
                        # 將日期設置為過去的日期
                        past_date = today - datetime.timedelta(days=random_count)
                        return past_date.strftime('%B %d, %Y')
                    elif template == 'YYYY_YY_TEMPLATE_PAST':
                        # 將日期設置為過去的日期
                        this_year = today.year
                        return str(this_year - 1) + '-' + str(this_year)[-2:]
                    return match.group(0)  # 如果無法識別模板，保持原樣
                
                # 使用正則表達式替換所有匹配項
                question = re.sub(template_pattern, replace_template, question)
            
            # 替換Apple產品名稱
            question = self._replace_apple_production(question)
            
            # 創建替換後的新項目
            new_item = item.copy()
            new_item['ques'] = question
            replaced_data.append(new_item)
        
        return replaced_data

    def _replace_apple_production(self, text):
        """替換問題中的Apple產品名稱"""
        LAST_IPHONE = 'iPhone 16 pro'
        LAST_IPHONE2 = 'iPhone 16'
        LAST_IPHONE2_COLOR = 'Pink'
        LAST_IPHONE3 = 'iPhone 15'
        LAST_APPLE_WATCH = 'Series 10'
        LAST_IOS_VERSION = 'iOS 18'
        LAST_MACBOOK_CHIP = 'M4 Pro'
        LAST_AIRPODS = 'AirPods (4th Generation)'
        STORAGE = '256GB'

        
        text = text.replace("LAST_IPHONE2", LAST_IPHONE2)
        text = text.replace("LAST_IPHONE2_COLOR", LAST_IPHONE2_COLOR)
        text = text.replace("LAST_IPHONE3", LAST_IPHONE3)
        text = text.replace("LAST_IPHONE", LAST_IPHONE)
        text = text.replace("LAST_APPLE_WATCH", LAST_APPLE_WATCH)
        text = text.replace("LAST_IOS_VERSION", LAST_IOS_VERSION)
        text = text.replace("LAST_MACBOOK_CHIP", LAST_MACBOOK_CHIP)
        text = text.replace("LAST_AIRPODS", LAST_AIRPODS)
        text = text.replace("STORAGE", STORAGE)
        
        
        return text

# 使用範例
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='')
    parser.add_argument('--output_file', type=str, default='')
    parser.add_argument("--min_days", type=int, default=30)
    parser.add_argument("--max_days", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    json_path = args.input_file
    replacer = TemplateReplacer(json_path)
    
    # 設定隨機種子以獲得可重現的結果，同時替換日期和Apple產品名稱
    replaced_data = replacer.replaced_dates(seed=args.seed, 
                                            min_days=args.min_days, 
                                            max_days=args.max_days)

    # 將結果保存為新的JSONL檔案
    output_path = args.output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in replaced_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n替換後資料已儲存至: {output_path}")

if __name__ == "__main__":
    main()