"""
Local Storage Manager for Chat History
Handles persistence of chat history with TTL
"""
import json
import time
from typing import List, Dict, Any
import streamlit as st


class ChatStorageManager:
    """Manages chat history persistence with TTL"""
    
    def __init__(self, ttl_minutes: int = 30):
        self.ttl_minutes = ttl_minutes
        self.storage_key = 'fraud_chatbot_history'
    
    def save_chat_history(self, chat_history: List[Dict[str, Any]]) -> None:
        """Save chat history to session state with TTL"""
        try:
            # Filter out processing messages and clean up data
            clean_history = []
            for chat in chat_history:
                if not chat.get('is_processing', False):
                    clean_chat = {
                        'question': chat.get('question', ''),
                        'response': chat.get('response'),
                        'timestamp': chat.get('timestamp'),
                        'saved_at': time.time()
                    }
                    clean_history.append(clean_chat)
            
            # Save to session state
            st.session_state.persistent_chat_history = {
                'data': clean_history,
                'saved_at': time.time(),
                'ttl_seconds': self.ttl_minutes * 60
            }
            
            # Also save to a backup key for recovery
            st.session_state.chat_history_backup = clean_history.copy()
            
        except Exception as e:
            st.error(f"Error saving chat history: {e}")
    
    def load_chat_history(self) -> List[Dict[str, Any]]:
        """Load chat history from session state, checking TTL"""
        try:
            if 'persistent_chat_history' not in st.session_state:
                return []
            
            stored_data = st.session_state.persistent_chat_history
            current_time = time.time()
            
            # Check if data has expired
            if current_time - stored_data['saved_at'] > stored_data['ttl_seconds']:
                # Clear expired data
                self.clear_expired_history()
                return []
            
            # Return valid data
            return stored_data['data']
            
        except Exception as e:
            st.error(f"Error loading chat history: {e}")
            return []
    
    def clear_expired_history(self) -> None:
        """Clear expired chat history from session state"""
        try:
            if 'persistent_chat_history' in st.session_state:
                del st.session_state.persistent_chat_history
            if 'chat_history_backup' in st.session_state:
                del st.session_state.chat_history_backup
        except Exception as e:
            st.error(f"Error clearing expired history: {e}")
    
    def is_expired(self) -> bool:
        """Check if current chat history has expired"""
        try:
            if 'persistent_chat_history' not in st.session_state:
                return True
            
            stored_data = st.session_state.persistent_chat_history
            current_time = time.time()
            
            return current_time - stored_data['saved_at'] > stored_data['ttl_seconds']
            
        except Exception:
            return True
    
    def get_remaining_ttl(self) -> int:
        """Get remaining TTL in seconds"""
        try:
            if 'persistent_chat_history' not in st.session_state:
                return 0
            
            stored_data = st.session_state.persistent_chat_history
            current_time = time.time()
            elapsed = current_time - stored_data['saved_at']
            remaining = stored_data['ttl_seconds'] - elapsed
            
            return max(0, int(remaining))
            
        except Exception:
            return 0
    
    def get_ttl_display(self) -> str:
        """Get human-readable TTL display"""
        remaining = self.get_remaining_ttl()
        if remaining <= 0:
            return "Expired"
        
        minutes = remaining // 60
        seconds = remaining % 60
        
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def initialize_from_session(self) -> None:
        """Initialize chat history from session state on app start"""
        try:
            # Load from persistent storage
            saved_history = self.load_chat_history()
            
            if saved_history and len(saved_history) > 0:
                # Restore to current chat history
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                
                # Add saved history to current session
                for chat in saved_history:
                    # Convert timestamp back to datetime if needed
                    if isinstance(chat.get('timestamp'), str):
                        from datetime import datetime
                        try:
                            chat['timestamp'] = datetime.fromisoformat(chat['timestamp'])
                        except:
                            chat['timestamp'] = datetime.now()
                    
                    st.session_state.chat_history.append(chat)
                
                st.success(f"📚 Restored {len(saved_history)} messages from previous session (TTL: {self.get_ttl_display()})")
            
        except Exception as e:
            st.error(f"Error initializing chat history: {e}")
    
    def cleanup_old_messages(self, max_messages: int = 50) -> None:
        """Keep only the most recent messages to prevent storage bloat"""
        try:
            if 'chat_history' in st.session_state:
                current_history = st.session_state.chat_history
                if len(current_history) > max_messages:
                    # Keep only the most recent messages
                    st.session_state.chat_history = current_history[-max_messages:]
                    st.info(f"🧹 Cleaned up chat history, keeping {max_messages} most recent messages")
        except Exception as e:
            st.error(f"Error cleaning up old messages: {e}")


def get_storage_manager() -> ChatStorageManager:
    """Get or create the storage manager instance"""
    if 'chat_storage_manager' not in st.session_state:
        st.session_state.chat_storage_manager = ChatStorageManager(ttl_minutes=30)
    return st.session_state.chat_storage_manager
