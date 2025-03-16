import os
import base64
import logging
from openai import AzureOpenAI
import pymupdf4llm
from pdfPaser_config import AzureConfig, PDFConfig

def setup_logging():
    """設定日誌記錄"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def Get_llm_description(
        llm_client, llm_model, image_blob, content_type, markdown=None
    ):
        #if prompt is None or prompt.strip() == "":
        #    prompt = "Write a detailed alt text for this image with less than 50 words."

        image_base64 = base64.b64encode(image_blob).decode("utf-8")
        data_uri = f"data:{content_type};base64,{image_base64}"
        
        # Create appropriate content based on whether markdown is provided
        if markdown is None:
            user_content = [
                {"type": "text", "text": ""},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_uri,
                    },
                }
            ]
        else:
            user_content = [
                {"type": "text", "text": markdown},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_uri,
                    },
                }
            ]

        messages = [
            {
                "role": "system",
                "content": """
                You are an assistant for generating image descriptions. The user will provide a segment of Markdown, which includes details about an image `{image}`. Based on the text provided in the Markdown, describe the content of the image. Please adhere to the following guidelines:

1. The generated language must match the language used in the content provided by the user. If the text is in Chinese, ensure whether it is Traditional or Simplified Chinese. If the user does not specify, use English.
2. Objectively describe the facts with concise and neutral wording.
3. Assume your reply is `{Your answer}`. Ensure that when the Markdown syntax `![{Your answer}]` is used to replace `{image}`, the meaning remains unchanged, and the Markdown syntax is valid.
"""
            },
            {
                "role": "user",
                "content": user_content
            }
        ]

        response = llm_client.chat.completions.create(
            model=llm_model, messages=messages
        )
        return response.choices[0].message.content


def Get_markdown_image_nearby(markdown, image_path):
    lines = markdown.split("\n")
    headers = []
    current_level = 0
    content = []
    last_header_index = 0
    
    for i, line in enumerate(lines):
        if line.startswith("#"):
            level = len(line.split()[0])  # Count number of # symbols
            while current_level >= level:
                if headers:
                    headers.pop()
                current_level -= 1
            headers.append(line.strip())
            current_level = level
            last_header_index = i
            
        if image_path in line:
            # Get content from last header to current line
            content = lines[last_header_index + 1 : i + 1]
            header_structure = " > ".join(headers) if headers else ""
            result = f"{header_structure}\n" + "\n".join(content) if headers else None
            return result.replace(image_path, "image") if result else None
            
    return None

def process_pdf(pdf_path: str, azure_config: AzureConfig, pdf_config: PDFConfig) -> None:
    """處理 PDF 文件"""
    try:
        client = AzureOpenAI(
            azure_endpoint=azure_config.endpoint,
            api_key=azure_config.api_key,
            api_version=azure_config.api_version
        )

        result = pymupdf4llm.to_markdown(
            pdf_path,
            write_images=True,
            image_path=pdf_config.image_save_path,
            image_format=pdf_config.image_format
        )

        for image_file in os.listdir(pdf_config.image_save_path):
            if image_file.endswith(pdf_config.image_format):
                image_path = os.path.join(pdf_config.image_save_path, image_file)
                markdown_content = Get_markdown_image_nearby(
                    result, 
                    f"{pdf_config.image_save_path}/{image_file}"
                )
                
                with open(image_path, "rb") as f:
                    image_blob = f.read()
                    content_type = f"image/{image_file.split('.')[-1]}"
                    description = Get_llm_description(
                        client, 
                        azure_config.model_name,
                        image_blob,
                        content_type,
                        markdown_content
                    )
                    
                    result = result.replace(
                        f"![]({pdf_config.image_save_path}/{image_file})",
                        f"![{description}]({pdf_config.image_save_path}/{image_file})"
                    )

        output_filename = os.path.splitext(os.path.basename(pdf_path))[0] + ".md"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(result)
            
        logging.info(f"PDF processed successfully: {output_filename}")
            
    except Exception as e:
        logging.error(f"Error processing PDF: {str(e)}")
        raise

def main():
    setup_logging()
    azure_config = AzureConfig()
    pdf_config = PDFConfig()
    
    pdf_path = r'E:\碩士\論文\Webvoyager-with-LangGraph\data\pdf\Arxiv操作說明.pdf'
    
    try:
        process_pdf(pdf_path, azure_config, pdf_config)
    except Exception as e:
        logging.error(f"Failed to process PDF: {str(e)}")

if __name__ == "__main__":
    main()