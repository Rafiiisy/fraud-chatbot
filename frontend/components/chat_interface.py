"""
Chat Interface Component
Handles the chat input and message display
"""
import streamlit as st
from datetime import datetime
from .storage_manager import get_storage_manager


class ChatInterface:
    def __init__(self):
        self.storage_manager = get_storage_manager()
        self.initialize_chat_history()
    
    def initialize_chat_history(self):
        """Initialize chat history from storage on first load"""
        if 'chat_history_initialized' not in st.session_state:
            # Check if there's existing chat history in storage
            saved_history = self.storage_manager.load_chat_history()
            
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
                
                # Show a subtle notification
                st.session_state.show_restore_message = True
            
            st.session_state.chat_history_initialized = True
    
    def render_chat_interface(self):
        # Check if we need to process a pending question
        if st.session_state.get('is_processing', False) and st.session_state.get('pending_question'):
            self._handle_pending_question()
        
        # Check if we need to make an API call (after user message is shown)
        if (st.session_state.chat_history and 
            len(st.session_state.chat_history) >= 2 and 
            st.session_state.chat_history[-1].get('is_processing', False) and
            st.session_state.chat_history[-2].get('question') and
            not st.session_state.chat_history[-1].get('response')):
            
            # Get the question from the user message
            question = st.session_state.chat_history[-2]['question']
            # Make API call
            self._process_question_with_spinner(question)
        
        # Question input
        question = st.text_area(
            "Enter your question about fraud data:",
            value=st.session_state.current_question,
            height=60,
            placeholder="e.g., 'How does the daily fraud rate fluctuate over time?'",
            help="Try one of the sample questions in the sidebar or ask your own question",
            key=f"question_input_{len(st.session_state.chat_history)}"
        )
        
        # Action buttons
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
        with col2:
            if st.button("Clear", use_container_width=True):
                st.session_state.current_question = ""
                st.session_state.response_data = None
                st.rerun()

        with col3:
            if st.button("Send", type="primary", use_container_width=True):
                if question.strip():
                    # Add user message immediately
                    user_message = {
                        "question": question,
                        "response": None,
                        "timestamp": datetime.now(),
                        "is_processing": False
                    }
                    st.session_state.chat_history.append(user_message)
                    
                    # Add assistant message with loading state
                    assistant_message = {
                        "question": "",
                        "response": None,
                        "timestamp": datetime.now(),
                        "is_processing": True
                    }
                    st.session_state.chat_history.append(assistant_message)
                    
                    # Clear input and rerun to show messages
                    st.session_state.current_question = ""
                    st.rerun()
                else:
                    st.error("Please enter a question first.")
        
       
    
    def process_question(self, question):
        """Process the user's question using the backend API"""
        # Import API client
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from api_client import get_api_client
        
        # Get API client
        api_client = get_api_client()
        
        # Check if backend is healthy
        if not api_client.health_check():
            st.error("⚠️ Backend API is not available. Please ensure the backend is running on http://localhost:5000")
            # Update the last assistant message with error
            st.session_state.chat_history[-1]["response"] = {
                "error": True,
                "message": "❌ Backend API not available",
                "details": "Please ensure the backend is running on http://localhost:5000"
            }
            st.session_state.chat_history[-1]["is_processing"] = False
            return
        
        # Process the question
        try:
            # Send question to backend
            response_data = api_client.ask_question(question)
            
            # Check if there was an error
            if response_data.get("status") == "error":
                # Update the last assistant message with error
                st.session_state.chat_history[-1]["response"] = {
                    "error": True,
                    "message": f"❌ Error: {response_data.get('error', 'Unknown error')}",
                    "details": response_data.get('message', 'No additional details')
                }
                st.session_state.chat_history[-1]["is_processing"] = False
            else:
                # Convert backend response to frontend format
                formatted_response = self._format_backend_response(response_data)
                
                # Update the last assistant message with response
                st.session_state.chat_history[-1]["response"] = formatted_response
                st.session_state.chat_history[-1]["is_processing"] = False
                
                # Set response data for display
                st.session_state.response_data = formatted_response
                
                # Store chart with response ID for easy access
                if 'chart' in formatted_response and formatted_response['chart'] is not None:
                    # Use same ID generation method as in HTML
                    import hashlib
                    response_hash = hashlib.md5(str(formatted_response).encode()).hexdigest()[:8]
                    response_id = f"response_{response_hash}"
                    st.session_state.available_charts[response_id] = formatted_response['chart']
                    st.session_state.latest_chart = formatted_response['chart']
                    print(f"DEBUG: Stored chart for response_id: {response_id}")
                    print(f"DEBUG: Available charts now: {list(st.session_state.available_charts.keys())}")
                else:
                    st.session_state.latest_chart = None
                    print("DEBUG: No chart to store")
            
        except Exception as e:
            # Handle any processing errors
            st.session_state.chat_history[-1]["response"] = {
                "error": True,
                "message": f"❌ Processing Error: {str(e)}",
                "details": "An error occurred while processing your question"
            }
            st.session_state.chat_history[-1]["is_processing"] = False
        
        # Save chat history to storage
        self.storage_manager.save_chat_history(st.session_state.chat_history)
        
        # Clean up old messages to prevent storage bloat
        self.storage_manager.cleanup_old_messages(max_messages=50)
        
        # Rerun to show the final response
        st.rerun()
    
    def _handle_pending_question(self):
        """Handle a pending question that needs to be processed"""
        question = st.session_state.pending_question
        
        # Add user message immediately to chat history (no processing flag)
        user_message = {
            "question": question,
            "response": None,  # No response yet
            "timestamp": datetime.now(),
            "is_processing": False  # User message is not processing
        }
        
        # Add to chat history immediately
        st.session_state.chat_history.append(user_message)
        
        # Add assistant message with loading state
        assistant_message = {
            "question": "",  # Empty for assistant
            "response": None,  # No response yet
            "timestamp": datetime.now(),
            "is_processing": True  # Assistant is processing
        }
        
        # Add assistant message to chat history
        st.session_state.chat_history.append(assistant_message)
        
        # Clear processing flags
        st.session_state.is_processing = False
        st.session_state.pending_question = None
        
        # Rerun to show both messages first
        st.rerun()
        
        # Make API call immediately after rerun
        self._process_question_with_spinner(question)
    
    def _process_question_with_spinner(self, question):
        """Process the question with a spinner and update the assistant message"""
        # Import API client
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from api_client import get_api_client
        
        # Get API client
        api_client = get_api_client()
        
        # Check if backend is healthy
        if not api_client.health_check():
            st.error("⚠️ Backend API is not available. Please ensure the backend is running on http://localhost:5000")
            # Update the last assistant message with error
            st.session_state.chat_history[-1]["response"] = {
                "error": True,
                "message": "❌ Backend API not available",
                "details": "Please ensure the backend is running on http://localhost:5000"
            }
            st.session_state.chat_history[-1]["is_processing"] = False
            return
        
        # Process the question
        try:
            # Send question to backend
            response_data = api_client.ask_question(question)
            
            # Check if there was an error
            if response_data.get("status") == "error":
                # Update the last assistant message with error
                st.session_state.chat_history[-1]["response"] = {
                    "error": True,
                    "message": f"❌ Error: {response_data.get('error', 'Unknown error')}",
                    "details": response_data.get('message', 'No additional details')
                }
                st.session_state.chat_history[-1]["is_processing"] = False
            else:
                # Convert backend response to frontend format
                formatted_response = self._format_backend_response(response_data)
                
                # Update the last assistant message with response
                st.session_state.chat_history[-1]["response"] = formatted_response
                st.session_state.chat_history[-1]["is_processing"] = False
                
                # Set response data for display
                st.session_state.response_data = formatted_response
                
                # Store chart with response ID for easy access
                if 'chart' in formatted_response and formatted_response['chart'] is not None:
                    # Use same ID generation method as in HTML
                    import hashlib
                    response_hash = hashlib.md5(str(formatted_response).encode()).hexdigest()[:8]
                    response_id = f"response_{response_hash}"
                    st.session_state.available_charts[response_id] = formatted_response['chart']
                    st.session_state.latest_chart = formatted_response['chart']
                    print(f"DEBUG: Stored chart for response_id: {response_id}")
                    print(f"DEBUG: Available charts now: {list(st.session_state.available_charts.keys())}")
                else:
                    st.session_state.latest_chart = None
                    print("DEBUG: No chart to store")
            
        except Exception as e:
            # Handle any processing errors
            st.session_state.chat_history[-1]["response"] = {
                "error": True,
                "message": f"❌ Processing Error: {str(e)}",
                "details": "An error occurred while processing your question"
            }
            st.session_state.chat_history[-1]["is_processing"] = False
        
        # Save chat history to storage
        self.storage_manager.save_chat_history(st.session_state.chat_history)
        
        # Clean up old messages to prevent storage bloat
        self.storage_manager.cleanup_old_messages(max_messages=50)
        
        # Rerun to show the final response
        st.rerun()
    
    def _format_backend_response(self, backend_response):
        """Convert backend response format to frontend format"""
        # Extract key information from backend response
        question_type = backend_response.get('question_type', 'general')
        
        # For document-based responses, use 'answer' instead of 'summary'
        if 'answer' in backend_response:
            explanation = backend_response.get('answer', 'No answer available')
        else:
            explanation = backend_response.get('summary', 'No summary available')
        
        insights = backend_response.get('insights', [])
        statistics = backend_response.get('statistics', {})
        data = backend_response.get('data', [])
        chart_data = backend_response.get('chart_data', {})
        
        # Create formatted response
        formatted_response = {
            "type": question_type,
            "title": self._get_title_for_question_type(question_type),
            "explanation": explanation,
            "content": insights,
            "metrics": self._format_statistics(statistics),
            "data": self._format_data(data),
            "chart": self._create_chart_from_data(data, chart_data, question_type),
            "sources": backend_response.get('sources', []),
            "sql_query": backend_response.get('sql_query', ''),
            "confidence": backend_response.get('confidence', 0),
            "metadata": backend_response.get('metadata', {})
        }
        
        # Handle chart data from backend if available
        if 'chart' in backend_response and backend_response['chart']:
            print(f"=== CHART DEBUG ===")
            print(f"Backend chart data: {backend_response['chart']}")
            chart = self._create_chart_from_data(
                data, 
                backend_response['chart'], 
                question_type
            )
            print(f"Created chart: {chart is not None}")
            if chart:
                print(f"Chart type: {type(chart)}")
            formatted_response["chart"] = chart
        
        return formatted_response
    
    def _get_title_for_question_type(self, question_type):
        """Get appropriate title based on question type"""
        titles = {
            'temporal_analysis': '📈 Fraud Rate Analysis',
            'merchant_analysis': '🏪 Merchant Fraud Analysis',
            'geographic_analysis': '🌍 Geographic Fraud Analysis',
            'value_analysis': '💰 Fraud Value Analysis',
            'fraud_methods': '📄 Fraud Methods Analysis',
            'system_components': '🔧 Fraud Detection System Components'
        }
        return titles.get(question_type, '🤖 Fraud Analysis')
    
    def _format_statistics(self, statistics):
        """Format statistics for display"""
        if not statistics:
            return {}
        
        formatted = {}
        for key, value in statistics.items():
            if isinstance(value, dict):
                # Handle nested dictionaries (like trend)
                if key == 'trend':
                    formatted[key] = f"{value.get('direction', 'unknown')} ({value.get('strength', 'unknown')})"
                else:
                    formatted[key] = str(value)
            elif isinstance(value, (int, float)):
                if 'rate' in key.lower():
                    formatted[key] = f"{value:.2%}"
                else:
                    formatted[key] = f"{value:.2f}"
            else:
                formatted[key] = str(value)
        
        return formatted
    
    def _format_data(self, data):
        """Format data for display"""
        if not data:
            return None
        
        try:
            import pandas as pd
            return pd.DataFrame(data)
        except Exception:
            return data
    
    def _create_chart_from_data(self, data, chart_data, question_type):
        """Create chart from data based on question type or backend chart configuration"""
        if not data and not chart_data:
            return None
        
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            import pandas as pd
            
            # If we have backend chart configuration, use it
            if chart_data and isinstance(chart_data, dict) and 'type' in chart_data:
                return self._create_chart_from_backend_config(chart_data)
            
            # Fallback to original logic for data-based chart creation
            if not data:
                return None
                
            df = pd.DataFrame(data)
            
            if question_type == 'temporal_analysis':
                # Line chart for temporal data
                if 'date' in df.columns and 'fraud_rate' in df.columns:
                    fig = px.line(
                        df, 
                        x='date', 
                        y='fraud_rate',
                        title=chart_data.get('title', 'Fraud Rate Over Time'),
                        labels={'fraud_rate': 'Fraud Rate', 'date': 'Date'}
                    )
                    fig.update_layout(height=400)
                    return fig
            
            elif question_type == 'merchant_analysis':
                # Bar chart for merchant data
                if 'fraud_rate' in df.columns:
                    x_col = chart_data.get('x_col', df.columns[0])
                    fig = px.bar(
                        df.head(10),  # Top 10 merchants
                        x='fraud_rate',
                        y=x_col,
                        orientation='h',
                        title=chart_data.get('title', 'Top Merchants by Fraud Rate'),
                        labels={'fraud_rate': 'Fraud Rate', x_col: 'Merchant'}
                    )
                    fig.update_layout(height=500)
                    return fig
            
            elif question_type == 'geographic_analysis':
                # Bar chart for geographic data
                if 'region' in df.columns and 'fraud_rate' in df.columns:
                    fig = px.bar(
                        df,
                        x='region',
                        y='fraud_rate',
                        title=chart_data.get('title', 'Fraud Rate by Region'),
                        labels={'fraud_rate': 'Fraud Rate', 'region': 'Region'},
                        color='region'
                    )
                    fig.update_layout(height=400)
                    return fig
            
            elif question_type == 'value_analysis':
                # Pie chart for value data
                if 'fraud_value' in df.columns:
                    labels_col = chart_data.get('labels_col', 'transaction_type')
                    values_col = chart_data.get('values_col', 'fraud_value')
                    fig = px.pie(
                        df,
                        values=values_col,
                        names=labels_col,
                        title=chart_data.get('title', 'Fraud Value Distribution')
                    )
                    fig.update_layout(height=400)
                    return fig
            
        except Exception as e:
            print(f"Error creating chart: {e}")
            return None
        
        return None
    
    def _create_chart_from_backend_config(self, chart_config):
        """Create Plotly chart from backend chart configuration"""
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            import pandas as pd
            
            chart_type = chart_config.get('type', 'bar_chart')
            data = chart_config.get('data', [])
            title = chart_config.get('title', 'Chart')
            
            if not data:
                return None
            
            df = pd.DataFrame(data)
            
            if chart_type == 'bar_chart':
                x_col = chart_config.get('x_col', df.columns[0])
                y_col = chart_config.get('y_col', df.columns[1])
                orientation = chart_config.get('orientation', 'vertical')
                
                if orientation == 'horizontal':
                    fig = px.bar(
                        df,
                        x=y_col,
                        y=x_col,
                        orientation='h',
                        title=title,
                        color=x_col if len(df) <= 10 else None
                    )
                else:
                    fig = px.bar(
                        df,
                        x=x_col,
                        y=y_col,
                        title=title,
                        color=x_col if len(df) <= 10 else None
                    )
                
                fig.update_layout(height=400)
                return fig
            
            elif chart_type == 'line_chart':
                x_col = chart_config.get('x_col', df.columns[0])
                y_col = chart_config.get('y_col', df.columns[1])
                
                fig = px.line(
                    df,
                    x=x_col,
                    y=y_col,
                    title=title,
                    markers=True
                )
                
                # Format y-axis for percentages if it looks like a rate
                if 'rate' in y_col.lower() or 'percentage' in y_col.lower():
                    fig.update_yaxes(tickformat='.2%')
                
                fig.update_layout(height=400)
                return fig
            
            elif chart_type == 'multi_line_chart':
                x_col = chart_config.get('x_col', 'time_period')
                y_col = chart_config.get('y_col', 'fraud_percentage')
                group_col = chart_config.get('group_col', 'period_type')
                
                fig = px.line(
                    df,
                    x=x_col,
                    y=y_col,
                    color=group_col,
                    title=title,
                    markers=True
                )
                
                # Format y-axis for percentages
                fig.update_yaxes(tickformat='.2%')
                fig.update_layout(height=400)
                return fig
            
            elif chart_type == 'comparison_chart':
                x_col = chart_config.get('x_col', df.columns[0])
                y_col = chart_config.get('y_col', df.columns[1])
                
                fig = px.bar(
                    df,
                    x=x_col,
                    y=y_col,
                    title=title,
                    color=x_col,
                    text=y_col
                )
                
                # Add value labels on bars
                fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
                fig.update_layout(height=400)
                return fig
            
            elif chart_type == 'pie_chart':
                labels_col = chart_config.get('labels_col', df.columns[0])
                values_col = chart_config.get('values_col', df.columns[1])
                
                fig = px.pie(
                    df,
                    values=values_col,
                    names=labels_col,
                    title=title
                )
                
                fig.update_layout(height=400)
                return fig
            
            else:
                # Default to bar chart for unknown types
                return self._create_chart_from_backend_config({
                    **chart_config,
                    'type': 'bar_chart'
                })
                
        except Exception as e:
            print(f"Error creating chart from backend config: {e}")
            return None
    
    def get_chat_html(self):
        """Get chat content as HTML string"""
        # Build the complete chat content as HTML
        chat_content = ""
        
        if st.session_state.chat_history:
            # Group messages by conversation pairs
            i = 0
            while i < len(st.session_state.chat_history):
                chat = st.session_state.chat_history[i]
                
                # User message (right side) - appears first
                if chat['question']:  # Only show if there's a question
                    user_message = chat['question'].replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
                    chat_content += f'<div class="chat-message user-message"><div class="user-label"><strong>👤 You</strong></div><div class="user-content">{user_message}</div></div>'
                
                # Check if next message is assistant response
                if i + 1 < len(st.session_state.chat_history):
                    assistant_chat = st.session_state.chat_history[i + 1]
                    
                    # Assistant response (left side) - appears second
                    if assistant_chat.get('is_processing', False):
                        # Show loading state
                        chat_content += f'<div class="chat-message assistant-message loading"><div class="assistant-label"><strong>🤖 Assistant</strong></div><div class="assistant-content"><div class="loading-animation">Thinking<span class="loading-dots">...</span></div></div></div>'
                    elif assistant_chat.get('response'):
                        # Show actual response or error
                        if assistant_chat['response'].get('error', False):
                            # Error state
                            error_message = assistant_chat['response']['message'].replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
                            error_details = assistant_chat['response']['details'].replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
                            chat_content += f'<div class="chat-message assistant-message error"><div class="assistant-label"><strong>🤖 Assistant</strong></div><div class="assistant-content"><div class="error-message">{error_message}</div><div class="error-details">{error_details}</div></div></div>'
                        else:
                            # Normal response
                            assistant_response = self.get_response_html(assistant_chat['response'])
                            chat_content += f'<div class="chat-message assistant-message"><div class="assistant-label"><strong>🤖 Assistant</strong></div><div class="assistant-content">{assistant_response}</div></div>'
                    else:
                        # No response yet (shouldn't happen with new logic)
                        chat_content += f'<div class="chat-message assistant-message"><div class="assistant-label"><strong>🤖 Assistant</strong></div><div class="assistant-content">Processing...</div></div>'
                    
                    # Skip the assistant message in next iteration
                    i += 2
                else:
                    # No assistant message yet, just user message
                    i += 1
                
                # Add separator between conversations
                if i < len(st.session_state.chat_history):
                    chat_content += '<hr style="border: 1px solid #333; margin: 1rem 0;">'
        else:
            chat_content = '<div class="empty-chat-state"><div style="text-align: center; width: 100%;"><p style="margin: 0; color: #cccccc; font-size: 1.1rem;">No conversations yet. Ask a question to get started!</p></div></div>'
        
        return chat_content
    
    def render_chat_container(self):
        """Render the scrollable chat container (legacy method)"""
        chat_content = self.get_chat_html()
        st.markdown(chat_content, unsafe_allow_html=True)
    
    def get_response_html(self, response_data):
        """Convert response data to HTML string"""
        if not response_data:
            return "No response available."
        
        html = ""
        
        # Response title
        title = str(response_data.get('title', '')).replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
        html += f"<strong>{title}</strong><br><br>"
        
        # Explanation - convert markdown to HTML
        explanation = str(response_data.get('explanation', ''))
        # Convert markdown to HTML
        explanation_html = self._markdown_to_html(explanation)
        html += f"{explanation_html}<br><br>"
        
        # Metrics (if available)
        if 'metrics' in response_data and response_data['metrics']:
            html += '<div style="display: flex; gap: 1rem; margin: 1rem 0;">'
            for key, value in response_data['metrics'].items():
                if isinstance(value, (int, float)):
                    if 'rate' in key.lower() or 'share' in key.lower():
                        display_value = f"{value:.1f}%"
                    else:
                        display_value = f"{value:.2f}"
                else:
                    display_value = str(value)
                
                display_value = display_value.replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
                key_clean = str(key).replace('_', ' ').title().replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
                html += f'<div style="background: #333; padding: 0.5rem; border-radius: 0.25rem; text-align: center; flex: 1;"><strong>{key_clean}</strong><br>{display_value}</div>'
            html += "</div>"
        
        # Content list (if available)
        if 'content' in response_data and response_data['content']:
            for item in response_data['content']:
                item_clean = str(item).replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
                html += f"{item_clean}<br>"
        
        # Chart indicator (if available) - Just show a subtle indicator
        if self.is_chart_generatable(response_data):
            html += '<div style="margin-top: 0.5rem; font-style: italic; color: #666; font-size: 0.9rem;">'
            html += '📊 Chart data available - use buttons below to generate visualization'
            html += '</div>'
        
        # Sources removed - no longer displayed in UI
        
        return html
    
    def is_chart_generatable(self, response_data):
        """Check if response has chart data available"""
        return 'chart' in response_data and response_data['chart'] is not None
    
    def _markdown_to_html(self, markdown_text):
        """Convert basic markdown to HTML"""
        if not markdown_text:
            return ""
        
        # Convert markdown to HTML
        html = markdown_text
        
        # Convert **bold** to <strong>bold</strong>
        import re
        html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
        
        # Convert *italic* to <em>italic</em>
        html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)
        
        # Convert bullet points to HTML list
        lines = html.split('\n')
        in_list = False
        result_lines = []
        
        for line in lines:
            if line.strip().startswith('- '):
                if not in_list:
                    result_lines.append('<ul>')
                    in_list = True
                # Remove the - and add list item
                content = line.strip()[2:].replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
                result_lines.append(f'<li>{content}</li>')
            else:
                if in_list:
                    result_lines.append('</ul>')
                    in_list = False
                # Escape HTML characters for regular text
                content = line.replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
                result_lines.append(content)
        
        if in_list:
            result_lines.append('</ul>')
        
        # Join lines and convert line breaks to <br>
        html = '<br>'.join(result_lines)
        
        return html
    
    def render_response_inline(self, response_data):
        """Render response content inline within chat container"""
        if not response_data:
            return
        
        # Response title
        st.markdown(f"**{response_data['title']}**")
        
        # Explanation
        st.markdown(response_data['explanation'])
        
        # Metrics (if available)
        if 'metrics' in response_data:
            cols = st.columns(len(response_data['metrics']))
            for i, (key, value) in enumerate(response_data['metrics'].items()):
                with cols[i]:
                    st.metric(
                        label=key.replace('_', ' ').title(),
                        value=value if isinstance(value, (int, float)) else f"{value:.1f}%" if 'rate' in key or 'share' in key else value
                    )
        
        # Chart (if available)
        if 'chart' in response_data:
            st.plotly_chart(response_data['chart'], use_container_width=True)
        
        # Content list (if available)
        if 'content' in response_data:
            for item in response_data['content']:
                st.markdown(item)
        
        # Data table (if available)
        if 'data' in response_data:
            st.dataframe(response_data['data'], use_container_width=True)
        
        # Sources removed - no longer displayed in UI
