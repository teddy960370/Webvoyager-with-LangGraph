
import logging
from typing import Dict, Any, Optional
import requests
from dataclasses import dataclass


@dataclass
class RagflowAPIConfig:
    """Configuration for RAGFlow API"""
    base_url: str
    chat_id: str 
    api_key: str

class RagflowAPI:
    def __init__(self, config: RagflowAPIConfig):
        self.config = config
        self.headers = {
            'Authorization': f'Bearer {config.api_key}',
            'Content-Type': 'application/json',
        }
        self.base_url = f"{config.base_url}/api/v1/chats/{config.chat_id}"

    def create_session(self, session_name: str = "new session") -> Optional[str]:
        """Create a new chat session"""
        try:
            response = requests.post(
                f"{self.base_url}/sessions",
                headers=self.headers,
                json={"name": session_name}
            )
            response.raise_for_status()
            return response.json()['data']['id']
        except requests.exceptions.RequestException as e:
            logging.error("Session creation failed: %s", str(e))
            return None

    def get_completion(self, query: str, session_id: str) -> Optional[str]:
        """Get completion from RAGFlow API"""
        try:
            response = requests.post(
                f"{self.base_url}/completions",
                headers=self.headers,
                json={
                    "question": query,
                    "stream": False,
                    "session_id": session_id
                }
            )
            response.raise_for_status()
            return response.json()['data']['answer']
        except requests.exceptions.RequestException as e:
            logging.error("Completion request failed: %s", str(e))
            return None
        
    def delete_session(self, session_id: str) -> Optional[str]:
        """Delete a chat session"""
        try:
            response = requests.delete(
                f"{self.base_url}/sessions/{session_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error("Session deletion failed: %s", str(e))
            return None