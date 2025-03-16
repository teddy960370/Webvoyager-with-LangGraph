from dataclasses import dataclass

@dataclass
class AzureConfig:
    """Azure OpenAI 設定"""
    endpoint: str = "https://AzureEndpoint.azure.com/"
    api_key: str = "Azure key"
    api_version: str = "2024-02-15-preview"
    model_name: str = "gpt-4o"

@dataclass
class RagflowConfig:
    """RAGFlow API 設定"""
    base_url: str = "http://localhost:9380"
    chat_id: str = "your-chat-id"
    api_key: str = "ragflow-key"

@dataclass
class PDFConfig:
    """PDF 處理相關設定"""
    image_save_path: str = "extracted_images"
    image_format: str = "png"
    output_file: str = "output.md"
